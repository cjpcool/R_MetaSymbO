import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import copy
import random
import re
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
from google import genai
from google.genai import types
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

from utils.llm_utils import (
    parse_graph,
    graph_to_text,
    construct_supervisor_input,  # kept for potential future use, but not called
    classify_nodes_with_geometry,
    find_closest_structure,
)
from modules.geom_autoencoder import GeomVAE, EncoderwithPredictionHead
from modules.submodules import LatticeNormalizer, DisentangledDenoise
from modules.ldm.ddim_disentangle import LatentDiffusionDDIM_GeomVAE
from modules.ldm.scheduler import DDPM_Scheduler
from visualization.vis import visualizeLattice
from model.symb_logi_ops import (
    pairwise_l2,
    sinkhorn_log,
    kl_gaussian_diag,
    node_logic_loss,
    poe,
    qoe,
    mixture_mm,
)
from utils.mat_utils import frac_to_cart_coords


ENTITY_EXTRACTION="""
You are a material scientist. Given a material claim, your goal is to extract the material object, i.e., chemical elements/smiles/molecular formula of the object, the conditions, e.g., temperatures, presures or any conditions in the claim, and properties, i.e., the property of the object claimed in the claim.
Moreover, output a design requirements for designing the specific object according to the claim, be as specific as possible.
Only output the result in the following format:
Object: xxxx
Conditions: xxxx
Properties: xxxx
Design requirements: xxxx
"""






INSTRUCTIONS_TRANSLATOR = """
You are a crystallographer and solid-state materials scientist with rich knowledge in superconductor, aloy, etc.

Task
-----
Given a single design requirement, propose the simplest traditional crystal scaffold that matches it.
Pick from canonical lattices and prototypes (e.g., SC, BCC, FCC, Diamond (C), Zincblende (GaAs), Wurtzite (ZnO), Perovskite (ABO3), Rocksalt (NaCl)).

Describe the scaffold as a periodic graph **with element types (atomic numbers Z)**:

Return **only** the following code block, with no extra text.

Format
------
~~~
Node number: <N>
Element Z (per node):
Z0
Z1
...
Z(N-1)

Node fractional coordinates:
(x0, y0, z0)
...
(xN-1, yN-1, zN-1)

# (Optional) Edges are ignored by the pipeline.
Edges:
(i0, j0)
...
(iM-1, jM-1)

Lattice lengths: [a, b, c]
Lattice angles: [alpha, beta, gamma]
~~~

Constraints
-----------
- Use **fractional** coordinates in [0,1).
- Keep N minimal while capturing the prototype.
- Choose realistic **element Z** for the prototype (e.g., NaCl: 11 & 17), but minimal stoichiometry is fine.
- Do not include any text outside the code block.
"""

MAX_NODE_NUM = 500  # crystals typically small protos

class CrystalGenAgents(nn.Module):
    """Crystal-ready revision of the original MetamatGenAgents.

    This phase removes **Supervisor Agent 3**. We rely on:
    - Agent 1 (Translator): proposes **crystal scaffolds** with element types (Z).
    - Agent 2 (Generator): autoencoder for crystals using **cartesian coords**; edges are built internally via `radius_graph`.
    """

    def __init__(
        self,
        root='../',
        ckpt_dir='/checkpoints/omat24_rattle2/',
        device='cuda',
        backbone='vae',
        designer_client='gpt-4o-mini',
        api_key='',
        evaluation_threshold=0.5,  # kept but unused
        max_evaluate_num=1,        # not used (no supervision loop)
        Generator=None,
        latent_dim=128,
        condition_dim: int = 1,
    ):
        super().__init__()
        self.designer_client = designer_client
        self.api_key = api_key
        self.backbone = backbone
        self.device = device
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.root = root
        self.ckpt_dir = ckpt_dir
        self.normalizer = LatticeNormalizer()
        self.evaluation_threshold = evaluation_threshold
        self.max_evaluate_num = max_evaluate_num

        
        self.is_disent_variational = False
        
        self.Translator = self._load_translator()
        self.Generator = self._load_generator() if Generator is None else Generator
        self.Predictor = None  # optional; not used for stopping criteria

    # -------------------- infra --------------------
    def init_normalizer(self, train_dataset):
        self.normalizer.fit(train_dataset.lengths, train_dataset.angles)

    def _load_translator(self):
        if 'gpt' in self.designer_client or 'o4-mini' in self.designer_client:
            client = OpenAI(api_key=self.api_key)
        elif 'gemini' in self.designer_client:
            client = genai.Client(api_key=self.api_key)
        return client

    def _load_generator(self, latent_dim=128, condition_dim=1, edge_sample_threshold=0.5):
        AEmodel = GeomVAE(
            self.normalizer,
            max_node_num=MAX_NODE_NUM,
            latent_dim=latent_dim,
            edge_sample_threshold=edge_sample_threshold,
            is_variational=True,
            is_disent_variational=False,
            is_condition=True,
            condition_dim=condition_dim,
            disentangle_same_layer=True,
        ).to(self.device)
        
        ae_ckpt = os.path.join(self.root, self.ckpt_dir, 'best_ae_model.pt')
        if os.path.exists(ae_ckpt):
            AEmodel.load_state_dict(torch.load(ae_ckpt, map_location=self.device))
        AEmodel.eval()
        self.normalizer = AEmodel.normalizer

        if self.backbone == 'LDM':
            DiffModel = DisentangledDenoise(
                latent_dim=latent_dim, time_emb_dim=128, is_condition=False, condition_dim=condition_dim
            )
            diff_ckpt = os.path.join(self.root, self.ckpt_dir, 'best_diff_model.pt')
            if os.path.exists(diff_ckpt):
                DiffModel.load_state_dict(torch.load(diff_ckpt, map_location=self.device))
            diffscheduler = DDPM_Scheduler(num_timesteps=1000, beta_start=1e-4, beta_end=2e-3, schedule_type='linear')
            LDM = LatentDiffusionDDIM_GeomVAE(
                AEmodel, DiffModel, scheduler=diffscheduler, lr_vae=1e-3, lr_diffusion=1e-3, device=self.device,
                is_condition=False, condition_dim=condition_dim
            )
            LDM.eval()
            return LDM
        return AEmodel


    def entity_extraction(self, input_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Extract material entities from the input text using LLM.
        Returns:
            A tuple of (object, conditions, properties, design requirements).
        """
        
        # Parse the LLM output into the four fields
        def _strip_fences(t: str) -> str:
            m = re.search(r"```(?:[\w+-]*)?\s*(.*?)```", t, re.S)
            return m.group(1).strip() if m else t

        def _extract_field(t: str, keys):
            if not t:
                return None
            keys = keys if isinstance(keys, (list, tuple)) else [keys]
            for k in keys:
                m = re.search(rf'^{re.escape(k)}\s*[:\-]\s*(.*)$', t, re.I | re.M)
                if m:
                    return m.group(1).strip()
            return None
        
        def _norm(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            x = x.strip()
            return x if x and x.lower() not in {"n/a", "none", "null", "-"} else None


        # Use the correct system instruction for entity extraction
        try:
            if 'gpt' in self.designer_client or 'o4-mini' in self.designer_client:
                response = self.Translator.responses.create(
                    model=self.designer_client,
                    instructions=ENTITY_EXTRACTION,
                    input=input_text,
                )
                output = response.output_text
            elif 'gemini' in self.designer_client:
                response = self.Translator.models.generate_content(
                    model=self.designer_client,
                    config=types.GenerateContentConfig(system_instruction=ENTITY_EXTRACTION),
                    contents=input_text,
                )
                output = response.text
        except Exception as e:
            warnings.warn(f'Entity extraction LLM call failed: {e}')
            return None, None, None, input_text

        text = _strip_fences(output.strip())

        obj = _extract_field(text, ["Object", "Material", "Composition"])
        cond = _extract_field(text, ["Conditions", "Condition", "Synthesis conditions"])
        prop = _extract_field(text, ["Properties", "Property"])
        req = _extract_field(text, ["Design requirements", "Design requirement", "Requirement", "Goal", "Target"])


        obj, cond, prop, req = map(_norm, (obj, cond, prop, req))
        if req is None:
            req = input_text.strip() or None
            
       
        self.object = obj
        self.condition = cond
        self.property = prop
        self.design_requirements = req
        
        return obj, cond, prop, req
        

    # -------------------- Translate design requirements to material --------------------
    def translate(self, input_text: str):
        """Agent-1: propose a **crystal** scaffold with element types (Z).
        Edges (if any) are ignored downstream.
        """
        while True:
            if 'gpt' in self.designer_client or 'o4-mini' in self.designer_client:
                response = self.Translator.responses.create(
                    model=self.designer_client,
                    instructions=INSTRUCTIONS_TRANSLATOR,
                    input=input_text,
                )
                output = response.output_text
            elif 'gemini' in self.designer_client:
                response = self.Translator.models.generate_content(
                    model=self.designer_client,
                    config=types.GenerateContentConfig(system_instruction=INSTRUCTIONS_TRANSLATOR),
                    contents=input_text,
                )
                output = response.text

            try:
                z, frac_coords, _edge_index_IGNORED, batch, lengths, angles, num_atoms = parse_graph(output)
            except ValueError as e:
                print(f"Error parsing graph: {e}\nOutput: {output}")
                continue
            break
        # convert to cartesian before feeding to AE
        cart_coords = frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)
        return z, cart_coords, batch, lengths, angles, num_atoms

    # -------------------- AE forward helpers --------------------
    @torch.no_grad()
    def reconstruct_from_input(self, node_type, cart_coords, batch, lengths, angles, num_atoms, num_samples):
        z = node_type.to(self.device)
        coords = cart_coords.to(self.device)
        batch = batch.to(self.device)
        lengths = lengths.to(self.device)
        angles = angles.to(self.device)
        num_atoms = num_atoms.to(self.device)

        lengths_normed, angles_normed = self.Generator.geomvae.normalizer(lengths, angles)
        semantic_latent, geo_coords_latent = self.Generator.geomvae.encoder(
            z, coords, batch, lengths_normed, angles_normed, num_atoms
        )
        node_num_logit, node_num_pred, lengths_pred, angles_pred, z_latent = self.Generator.geomvae.sample_lattice(
            num_samples=num_samples, z_latent=semantic_latent, device=self.device, random_scale_latent=1.0
        )
        return self.Generator.sample_ddim(
            num_samples=num_samples, ddim_steps=50, eta=1e-5, z_lattice=z_latent, z_g=geo_coords_latent, batch=batch, is_recon=True
        )

    # -------------------- Agent-1 single-shot (no Supervisor) --------------------
    def get_scaffold(self, input_text: str, visualize_results: bool = False):
        z, cart_coords, batch, lengths, angles, num_atoms = self.translate(input_text)
        if visualize_results:
            visualizeLattice(cart_coords.cpu().numpy(), None, title='Agent-1 scaffold (coords only)')
        return z, cart_coords, batch, lengths, angles, num_atoms

    # -------------------- 2 (generation/logic) --------------------
    def optimize_geo_latent(
        self,
        latent,
        batch,
        target_latent,
        batch_target,
        lr=0.01,
        num_steps=300,
        eps=1e-8,
        logic_mode='mix',
        lam=0.5,
        optimize_network_params=False,
        lam_node=1.0,
        lam_keep=1.0,
        lam_prior=1e-4,
        thr=0.1,
    ):
        print('optimizing geo latent (coords-only, no edges)...')
        AEmodel = copy.deepcopy(self.Generator)

        latent0 = latent.clone().detach().requires_grad_(True)
        params = [{'params': [latent0]}]
        if optimize_network_params:
            train_modules = [AEmodel.proj_in_semantic, AEmodel.proj_in_pos]
            for m in train_modules:
                m.requires_grad_(True)
                params.append({'params': m.parameters()})
        else:
            for p in AEmodel.parameters():
                p.requires_grad_(False)
        opt = torch.optim.Adam(params, lr=lr)

        with torch.no_grad():
            if self.is_disent_variational:
                z_p_t, z_s_t = AEmodel.disentangle_latent(target_latent, batch=batch_target)
                _, mu_sem_t0, log_sem_t0 = AEmodel.reparameterize(z_s_t, 'semantic', noise_scale=1.0)
                _, mu_node_t, log_node_t = AEmodel.reparameterize(z_p_t, 'coords', noise_scale=1.0)
                z_p_r, _ = AEmodel.disentangle_latent(latent, batch=batch)
                _, mu_node_old, log_node_old = AEmodel.reparameterize(z_p_r, 'coords', noise_scale=1.0)
            else:
                return latent.detach(), AEmodel
            
        for step in range(num_steps):
            opt.zero_grad()
            z_p, z_s = AEmodel.disentangle_latent(latent0, batch=batch)
            _, mu_sem, log_sem = AEmodel.reparameterize(z_s, 'semantic', noise_scale=1.0)
            _, mu_node, log_node = AEmodel.reparameterize(z_p, 'coords', noise_scale=1.0)

            if logic_mode == 'int':
                mu_sem_t, log_sem_t = poe(mu_sem, log_sem, mu_sem_t0, log_sem_t0)
            elif logic_mode == 'neg':
                mu_sem_t, log_sem_t = gaussian_negation(mu_sem, log_sem, mu_sem_t0, log_sem_t0, alpha=1.0, beta=0.01, eps=eps)
            else:
                mu_sem_t, log_sem_t = mixture_mm(mu_sem, log_sem, mu_sem_t0, log_sem_t0, lam=lam)

            L_sem = kl_gaussian_diag(mu_sem, log_sem, mu_sem_t.detach(), log_sem_t.detach()).mean()

            # coords-only logic & keep-original
            cost = pairwise_l2(mu_node.detach(), mu_node_t.detach())
            cost = cost / (cost.max() + 1e-8)
            P = sinkhorn_log(cost, epsilon=0.1)

            L_node = node_logic_loss(P, mu_node, log_node, mu_node_t.detach(), log_node_t.detach(), logic_mode=logic_mode, lam=lam, thr=thr)
            L_node_keep = node_logic_loss(P, mu_node, log_node, mu_node_old.detach(), log_node_old.detach(), logic_mode='keep', lam=1.0, thr=thr)

            loss = L_sem + lam_node * (L_node + lam_keep * L_node_keep) + lam_prior * latent0.pow(2).mean()
            loss.backward()
            opt.step()
        return latent0.detach(), AEmodel

    def optimize_lattice_latent(
        self,
        latent_src,
        latent_tgt,
        condition,
        lr=0.01,
        num_steps=30,
        logic_mode='union',
        lam_mix=0.5,
        eps=1e-8,
    ):
        AE = self.Generator
        print('optimizing lattice latent...')
        with torch.no_grad():
            _, mu_tgt, log_tgt = AE.reparameterize(latent_tgt)
            _, mu_init, log_init = AE.reparameterize(latent_src)
            if logic_mode == 'int':
                mu_teacher, log_teacher = poe(mu_init, log_init, mu_tgt, log_tgt)
            elif logic_mode == 'neg':
                mu_teacher, log_teacher = gaussian_negation(mu_init, log_init, mu_tgt, log_tgt, alpha=1.0, beta=0.01, eps=eps)
            elif logic_mode == 'mix':
                mu_teacher, log_teacher = mixture_mm(mu_init, log_init, mu_tgt, log_tgt, lam=lam_mix)
            else:
                raise ValueError('logic_mode must be int/neg/mix')
            mu_teacher = mu_teacher.detach()
            log_teacher = log_teacher.detach()

        latent = latent_src.clone().detach().requires_grad_(True).to(self.device)
        opt = torch.optim.Adam([latent], lr=lr)
        for step in range(num_steps):
            opt.zero_grad()
            _, mu_cur, log_cur = AE.reparameterize(latent)
            kl_loss = kl_gaussian_diag(mu_cur, log_cur, mu_teacher, log_teacher, eps=eps).mean()
            kl_loss.backward()
            opt.step()
        return latent.detach(), mu_cur.detach(), log_cur.detach()

    # -------------------- 2↔1 fusion (no edges) --------------------
    def collaborate_between_agents_12(
        self,
        batch_data,
        scaffold,
        logic_mode='mix',
        num_steps_lattice=30,
        num_steps_geo=300,
        optimize_network_params=False,
        thresh=0.5,   # kept for potential future fusion variants
        mix_lam=0.1,
        condition=None,
        lam_node=1.0,
        lam_keep=1.0,
        lam_prior=1e-4,
    ):
        self.Generator.eval()
        device = self.device

        # 1) encode scaffold (with node types Z); coords already cartesian
        z_pre, coords_pre, batch_pre, lengths_pre, angles_pre, num_atoms_pre = scaffold
        z_pre = torch.tensor(z_pre).to(device)
        coords_pre = torch.tensor(coords_pre).to(device)
        batch_pre = torch.tensor(batch_pre).to(device)
        lengths_pre = torch.tensor(lengths_pre).to(device)
        angles_pre = torch.tensor(angles_pre).to(device)
        num_atoms_pre = torch.tensor(num_atoms_pre).to(device)
        edge_index_pre = radius_graph(coords_pre, r=5.0)


        lengths_pre, angles_pre = self.normalizer(lengths_pre, angles_pre)
        with torch.no_grad():
            lattice_latent_pre, geo_latent_pre = self.Generator.encode(
                z_pre, coords_pre, edge_index_pre, batch_pre, lengths_pre, angles_pre, num_atoms_pre
            )

        # 2) encode dataset example (convert frac→cart first)
        z = batch_data.node_type.to(device)
        frac = batch_data.frac_coords.to(device)
        lengths = batch_data.lengths.to(device)
        angles = batch_data.angles.to(device)
        batch = batch_data.batch.to(device)
        num_atoms = batch_data.num_atoms.to(device)
        # cart = frac_to_cart_coords(frac, lengths, angles, num_atoms)
        cart = batch_data.cart_coords.to(device)
        condition = (batch_data.y if condition is None else condition).to(device)
        edge_index = batch_data.edge_index.to(device)

        lengths, angles = self.normalizer(lengths, angles)
        lattice_latent, geo_latent = self.Generator.encode(z, cart, edge_index, batch, lengths, angles, num_atoms)

        if logic_mode == 'union':
            optimized_geo_latent, newAEmodel = self.optimize_geo_latent(
                geo_latent,
                batch,
                geo_latent_pre,
                batch_pre,
                lr=0.01,
                num_steps=num_steps_geo,
                logic_mode=logic_mode,
                lam=mix_lam,
                optimize_network_params=optimize_network_params,
                thr=0.5,
                lam_node=lam_node,
                lam_keep=lam_keep,
                lam_prior=lam_prior,
            )
            optimized_lattice_latent = lattice_latent
        else:
            optimized_lattice_latent, _, _ = self.optimize_lattice_latent(
                lattice_latent, lattice_latent_pre, condition, lr=0.01, num_steps=num_steps_lattice, logic_mode=logic_mode, lam_mix=0.5
            )
            optimized_geo_latent, newAEmodel = self.optimize_geo_latent(
                geo_latent,
                batch,
                geo_latent_pre,
                batch_pre,
                lr=0.01,
                num_steps=num_steps_geo,
                logic_mode=logic_mode,
                lam=mix_lam,
                optimize_network_params=optimize_network_params,
                thr=0.2,
                lam_node=lam_node,
                lam_keep=lam_keep,
                lam_prior=lam_prior,
            )

        # 3) decode (coords-only)
        out = newAEmodel.decode_from_encoded(
            new_geo_latent=optimized_geo_latent,
            new_lattice_latent=optimized_lattice_latent,
            batch=batch,
            condition=condition,
            random_scale_latent=1.0,
            random_scale_geo=1.0,
        )
        # Support either legacy (with edges) or coords-only
        if len(out) == 6:
            node_num_pred, lengths_pred, angles_pred, coords_pred_list, _edge_list_IGNORED, y_pred = out
        else:
            node_num_pred, lengths_pred, angles_pred, coords_pred_list, y_pred = out

        # For this phase, skip OT merge with scaffold; directly use generated coords
        coords_all = coords_pred_list[0].detach()
        final_node_num = coords_all.shape[0]

        # --- assign element types Z (no edges) ---
        z_scaffold = z_pre.detach().view(-1)
        z_data = z.detach().view(-1)
        fill_Z = int(torch.mode(z_data).values.item()) if z_data.numel() > 0 else 14  # default Si
        if final_node_num <= z_scaffold.numel():
            z_all = z_scaffold[:final_node_num].clone()
        else:
            pad = torch.full((final_node_num - z_scaffold.numel(),), fill_Z, dtype=z_scaffold.dtype, device=z_scaffold.device)
            z_all = torch.cat([z_scaffold, pad], dim=0)

        return final_node_num, lengths_pred.detach(), angles_pred.detach(), coords_all.detach(), z_all.detach()

    # -------------------- End-to-end (no Supervisor) --------------------
    def collaborative_end_to_end_generation(
        self,
        dataset,
        input_text,
        logic_mode='union',
        num_steps_lattice=30,
        num_steps_geo=300,
        max_collaboration_num=1,  # single shot per your instruction
        optimize_network_params=False,
        mix_lam=0.2,
        condition=None,
        select_max_node_num=30,
        verbose=False,
        save_dir=None,
        save_index=None,
    ):
        indices = [i for i, d in enumerate(dataset) if d.num_atoms <= select_max_node_num and d.num_edges <= select_max_node_num * 2]
        select_init_data = dataset[indices]

        if condition is not None:
            condition = condition.to(self.device)

        for j in range(max_collaboration_num):
            print(f'====================> Generate trial index:{j}/{max_collaboration_num}')
            z, cart_coords, batch, lengths, angles, num_atoms = self.get_scaffold(input_text, visualize_results=verbose)
            if verbose:
                visualizeLattice(cart_coords.cpu().numpy(), None, title=f'scaffold iter {j}')

            scaffold = (z, cart_coords, batch, lengths, angles, num_atoms)
            batch_data = next(iter(DataLoader(select_init_data, batch_size=1, shuffle=True)))

            final_node_num, lengths_pred, angles_pred, coords_gen, z_gen = self.collaborate_between_agents_12(
                batch_data,
                scaffold,
                logic_mode=logic_mode,
                num_steps_lattice=num_steps_lattice,
                num_steps_geo=num_steps_geo,
                optimize_network_params=optimize_network_params,
                mix_lam=mix_lam,
                condition=condition,
                lam_node=1.0,
                lam_keep=1.0,
                lam_prior=1e-4,
            )

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                idx = random.randint(0, 10000) if save_index is None else save_index
                out = os.path.join(save_dir, f'{logic_mode}_{j}_{idx}.npz')
                np.savez(
                    out,
                    atom_types=z_gen.detach().cpu().numpy(),
                    lengths=lengths_pred.detach().cpu().view(-1).numpy(),
                    angles=angles_pred.detach().cpu().view(-1).numpy(),
                    cart_coords=coords_gen.detach().cpu().numpy(),
                    prop_list=(None if condition is None else condition.detach().cpu().numpy()),
                )

        return z_gen, coords_gen, lengths_pred, angles_pred, condition

    # -------------------- (Revised) Agent-1&2 entry (no Supervisor) --------------------
    def agent12_generation(
        self,
        dataset,
        input_text,
        logic_mode='union',
        num_steps_lattice=30,
        num_steps_geo=300,
        j=0,
        optimize_network_params=False,
        mix_lam=0.2,
        condition=None,
        select_max_node_num=30,
        verbose=False,
        save_dir=None,
    ):
        """Generate **crystal materials** using Agent-1 scaffold + Agent-2 latent fusion.

        This phase **does not** use Supervisor Agent 3. No LLM scoring or looped refinements.
        """
        indices = [i for i, d in enumerate(dataset) if d.num_atoms <= select_max_node_num and d.num_edges <= select_max_node_num * 2]
        select_init_data = dataset[indices]
        batch_data = next(iter(DataLoader(select_init_data, batch_size=1, shuffle=True)))

        print(f'====================> Generate trial {j}')
        z, cart_coords, batch, lengths, angles, num_atoms = self.get_scaffold(input_text, visualize_results=verbose)
        if verbose:
            visualizeLattice(cart_coords.cpu().numpy(), None, title=f'scaffold iter {j}')

        scaffold = (z, cart_coords, batch, lengths, angles, num_atoms)

        final_node_num, lengths_pred, angles_pred, coords_gen, z_gen = self.collaborate_between_agents_12(
            batch_data,
            scaffold,
            logic_mode=logic_mode,
            num_steps_lattice=num_steps_lattice,
            num_steps_geo=num_steps_geo,
            optimize_network_params=optimize_network_params,
            mix_lam=mix_lam,
            condition=condition,
            lam_node=1.0,
            lam_keep=1.0,
            lam_prior=1e-4,
        )

        if save_dir is not None:
            print(f"Saving the generated structure to {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            lattice_name = os.path.join(save_dir, f'{logic_mode}_{j}.npz')
            np.savez(
                lattice_name,
                atom_types=z_gen.detach().cpu().numpy(),
                lengths=lengths_pred.detach().cpu().view(-1).numpy(),
                angles=angles_pred.detach().cpu().view(-1).numpy(),
                cart_coords=coords_gen.detach().cpu().numpy(),
                prop_list=(None if condition is None else condition.detach().cpu().numpy()),
            )

        return z_gen, coords_gen, lengths_pred, angles_pred, condition
