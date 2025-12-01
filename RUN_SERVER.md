# Tutorial for access to fastapi servers
After I started the server in the open port, it is easy to access and run the model via curl.
## Output Json format example:
~~~
{"gen_path":"_gens/gen_union.npz","optimize":{"uma":{"energy_eV":-3.819211809084268,"traj":"./_mdopt/best.traj","xyz":"./_mdopt/best.xyz"},"dft":{"dipole_vec_D":[1.0695197423693514e-05,2.45837361596557e-05,-3.267579437249424e-05],"dipole_D":4.226647453210251e-05,"energy_hartree":-2020.995943482014,"energy_eV":-54994.101219664015,"homo_eV":-3.0082,"lumo_eV":-2.6833,"gap_eV":0.32489999999999997,"mulliken_block":"   0 Al:    0.000000    1.000000\n   1 Zn:    0.000000   -0.000000\nSum of atomic charges         :    0.0000000\nSum of atomic spin populations:    1.0000000","forces":[[-4.666552588461119e-07,-1.077446571746841e-06,1.4285050237735525e-06],[4.4351532865539554e-07,1.0286984521450653e-06,-1.3677755625396948e-06]]},"summary_path":null}}
~~~
In this output, you may need to query keys "uma" and "dft".


## Pipeline Running
~~~
curl -X POST http://zhoulab-1.cs.vt.edu:5560/pipeline -H "Content-Type: application/json" -d '{
    "claims": "Al20Zn80 at 870K is a solid at equilibrium",
    "ckpt_dir": "./checkpoints/omat24_rattle2",
    "use_llm": true,
    "save_dir": "./_gens",
    "no_generator": true,
    "api_key": "openai api key",
    "no-generator": true,

    "gen_path": "./_gens/gen_union.npz",
    "outdir": "./_mdopt",
    "run_dft": true,
    "uma_ckpt": "/home/grads/jianpengc/projects/materials/R_MetaSymbO/checkpoints/uma-s-1p1.pt",
    "loops": 2,
    "preset": "sprint" / "quick" / "standard",
    "orca_command": "/home/grads/jianpengc/orca_6_1_0/orca",
    "orca_maxcore": 4000,
    "orca_nprocs": 8
  }'
~~~


## Step by Step Running
Command for Step 1:

~~~
curl -X POST http://localhost:5560/generate \
  -H "Content-Type: application/json" \
  -d '{
    "claims": "Al20Zn80 at 870K is a solid at equilibrium",
    "ckpt_dir": "./checkpoints/omat24_rattle2",
    "use_llm": true,
    "save_dir": "./_gens",
    "no_generator": true,
    "api_key": "openai api key",
    "uma"
  }'
  ~~~

Command for Step 2 and 3 (uma optimize + dft computation):
~~~
curl -X POST http://localhost:5560/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "gen_path": "./_gens/gen_union.npz",
    "outdir": "./_mdopt",
    "run_dft": true,
    "ckpt": "/home/grads/jianpengc/projects/materials/R_MetaSymbO/checkpoints/uma-s-1p1.pt",
    "loops": 2,
    "preset": ["quick","sprint", "standard", "thorough"],
    "orca_command": "/home/grads/jianpengc/orca_6_1_0/orca",
    "orca_maxcore": 4000,
    "orca_nprocs": 8
  }'
~~~

