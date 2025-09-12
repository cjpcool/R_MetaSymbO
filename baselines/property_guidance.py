import os
import sys
import argparse
import torch
import re
import numpy as np
import concurrent.futures
import time
from tqdm import tqdm

INIT_PROMPT = "Design a valid and diverse structure, ensure it satisfies symmetry, and periodicity."

# Add path to allow imports from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Import the API functions from api.py
from api import generate_with_api


INSTRUCTIONS_TRANSLATOR = """
You are a metamaterial scientist specializing in structural design and mechanical characterization. You have expert knowledge of canonical 3‑D architectures (octet‑truss, BCC, SC, Kelvin cell, Diamond, TPMS, etc.) and their typical mechanical responses.

Task
-----

Given a single *design requirement*, your task is to generate a possible memtamaterial structures, described as a graph. The graph should be defined by:

- **Nodes** — 3‑D fractional coordinates.  
- **Edges** — pairs of node indices.
- **Lattice Lengths** - lengths of the unit cell in 3D.
- **Lattice Angles** - angles of the unit cell in 3D.

Output the graph in a code block exactly as shown below; provide **no additional text, commentary, or reasoning**.

Input
-----

Design prompt (free text).

Output format
-------------
~~~
Node number: <N>
Node coordinates:
(x1, y1, z1)
...
(xN, yN, zN)

Edges:
(i0, j0)
...
(iM, jM)

Lattice Lengths:
(L1, L2, L3)

Lattice Angles:
(A1, A2, A3)
~~~

Constraints
-----------

- Return the output *only* in the specified layout and code‑block format.  
- Do not include any other information.
"""


# Default worker counts for different model types
DEFAULT_WORKERS = {
    "openai": 4,
    "gemini": 8,
    "claude": 2,
    "llama": 4,
    "qwen": 8,
    "deepseek": 32,
}

MAX_NODE_NUM = 100

def parse_graph_llm(text: str):
    """
    Parse a node/edge description like the prompt example and return
    coords  – Tensor[N, 3]  (float32)
    edges   – Tensor[M, 2]  (long)
    """
    # --- split the text into its three sections -----------------------------
    #     1) header with node number (optional for parsing)
    #     2) node coordinates lines
    #     3) edge lines
    #
    # find where the coordinates section starts and where the edges start
    coord_start = text.index("Node coordinates:") + len("Node coordinates:")
    edge_start  = text.index("Edges:")

    lengths_start = text.index("Lattice Lengths:")
    angles_start = text.index("Lattice Angles:")
    
    coord_block = text[coord_start:edge_start].strip().splitlines()
    edge_block  = text[edge_start + len("Edges:"):lengths_start].strip().splitlines()
    lengths_block  = text[lengths_start + len("Lattice Lengths:"):angles_start].strip().splitlines()
    angles_block  = text[angles_start + len("Lattice Angles:"):].strip().splitlines()
    
    # --- helper: grab all numbers in a line ---------------------------------
    num_pat = re.compile(r'[-+]?\d*\.\d+|\d+')   # floats or ints

    # Parse coordinates -------------------------------------------------------
    coords = [
        [float(x) for x in num_pat.findall(line)]
        for line in coord_block if num_pat.search(line)
    ]
    coords = torch.tensor(coords, dtype=torch.float32)  # → [N, 3]

    # Parse edges -------------------------------------------------------------
    edges = [
        [int(float(x)) for x in num_pat.findall(line)]
        for line in edge_block if num_pat.search(line)
    ]
    edges = torch.tensor(edges, dtype=torch.long)       # → [M, 2]
    

    # Construct other elements:z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms,
    num_atoms = torch.LongTensor([coords.shape[0]])
    edge_index = edges.T
    try:
        node_labels = classify_nodes_with_geometry(coords, edge_index)
        z = torch.LongTensor(torch.argmax(node_labels,dim=-1)+1)
    except:
        Warning("No geometry information, using default node labels")
        z = torch.ones(coords.shape[0], dtype=torch.long)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    lengths = [
        [float(x) for x in num_pat.findall(line)]
        for line in lengths_block if num_pat.search(line)
    ]
    angles = [
        [float(x) for x in num_pat.findall(line)]
        for line in angles_block if num_pat.search(line)
    ]
    lengths = torch.FloatTensor([[1,1,1]])
    angles = torch.FloatTensor([[90,90,90]])

    return z, coords, edge_index, batch, lengths, angles, num_atoms

def parse_args():
    parser = argparse.ArgumentParser(description="Generate diverse metamaterial structures with property guidance.")
    
    # Input and output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results.")
    parser.add_argument("--num_structures", type=int, default=121,
                        help="Number of structures to generate.")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="llama-4-maverick",
                        help="Model to use for generation.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (higher values = more randomness).")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for processing. Default is model-dependent.")
    
    # API keys
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key (will use OPENAI_API_KEY env var if not provided).")
    parser.add_argument("--gemini_api_key", type=str, default=None,
                        help="Gemini API key (will use GEMINI_API_KEY env var if not provided).")
    parser.add_argument("--anthropic_api_key", type=str, default=None,
                        help="Claude API key (will use ANTHROPIC_API_KEY env var if not provided).")
    parser.add_argument("--lambda_api_key", type=str, default=None,
                        help="Lambda API key for Llama models (will use LAMBDA_API_KEY env var if not provided).")
    parser.add_argument("--qwen_api_key", type=str, default=None,
                        help="Qwen API key (will use QWEN_API_KEY env var if not provided).")
    parser.add_argument("--deepseek_api_key", type=str, default=None,
                        help="DeepSeek API key (will use DEEPSEEK_API_KEY env var if not provided).")
    
    # Conditionals
    parser.add_argument("--conditional", action="store_true",
                        help="Enable conditional generation based on properties.")
    parser.add_argument("--max_node_num", type=int, default=100,
                        help="Maximum number of nodes per structure.")
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", 
                                     f"{args.model}_property_guidance")
    
    return args

def generate_structure(index, model_name, model_type, temperature, max_tokens, init_prompt, generated_history="", condition=None):
    """Generate a single structure with the specified model"""
    
    # Prepare the input text
    if condition is not None:
        young = condition[:3]
        shear = condition[3:6]
        poisson = condition[6:]
        input_text = init_prompt + f" Properties: Young's modulus: {young}, Shear modulus: {shear}, Poisson ratio: {poisson}."
    else:
        input_text = init_prompt
    
    # If we have history, add it to prevent duplicates
    if generated_history:
        input_text += "\nAlready generated structures, do not generate them again:" + generated_history
    
    # Format conversation for API call
    conversation = [
        {"role": "system", "content": INSTRUCTIONS_TRANSLATOR},
        {"role": "user", "content": input_text}
    ]
    
    # Get model response using the generate_with_api function
    output, token_count = generate_with_api(
        model_type=model_type,
        model=model_name,
        conversation=conversation,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Parse the generated structure
    try:
        z, coords, edge_index, batch, lengths, angles, num_atoms = parse_graph_llm(output)
        return {
            "index": index,
            "output": output,
            "token_count": token_count,
            "z": z,
            "coords": coords,
            "edge_index": edge_index,
            "batch": batch,
            "lengths": lengths,
            "angles": angles,
            "num_atoms": num_atoms,
            "success": True
        }
    except Exception as e:
        print(f"Error parsing structure {index}: {str(e)}")
        return {
            "index": index,
            "output": output,
            "token_count": token_count,
            "success": False,
            "error": str(e)
        }

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set API keys from environment variables with fallback to command line arguments
    openai_api_key = os.environ.get("OPENAI_API_KEY", args.openai_api_key)
    gemini_api_key = os.environ.get("GEMINI_API_KEY", args.gemini_api_key)
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", args.anthropic_api_key)
    lambda_api_key = os.environ.get("LAMBDA_API_KEY", args.lambda_api_key)
    qwen_api_key = os.environ.get("QWEN_API_KEY", args.qwen_api_key)
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", args.deepseek_api_key)
    
    # Determine model type
    model_name = args.model
    is_openai_model = args.model.startswith("gpt") or args.model.startswith("o4") or args.model.startswith("o3") or args.model.startswith("o1")
    is_gemini_model = args.model.startswith("gemini")
    is_claude_model = args.model.startswith("claude")
    is_deepseek_model = args.model.startswith("deepseek")
    is_qwen_model = args.model.startswith("qwen") or args.model.startswith("qwq")
    is_llama_model = args.model.startswith("llama")
    
    if is_openai_model:
        model_type = "openai"
    elif is_gemini_model:
        model_type = "gemini"
    elif is_claude_model:
        model_type = "claude"
    elif is_deepseek_model:
        model_type = "deepseek"
    elif is_qwen_model:
        model_type = "qwen"
    elif is_llama_model:
        model_type = "llama"
    else:
        print(f"Error: Unsupported model '{model_name}'.")
        return
    
    # Set appropriate API keys based on model type
    if model_type == "openai" and openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif model_type == "gemini" and gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    elif model_type == "claude" and anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    elif model_type == "llama" and lambda_api_key:
        os.environ["LAMBDA_API_KEY"] = lambda_api_key
        os.environ["OPENAI_API_KEY"] = lambda_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.lambda.ai/v1"
    elif model_type == "qwen" and qwen_api_key:
        os.environ["QWEN_API_KEY"] = qwen_api_key
        os.environ["OPENAI_API_KEY"] = qwen_api_key
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif model_type == "deepseek" and deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
        os.environ["OPENAI_API_KEY"] = deepseek_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    else:
        print(f"Error: API key required for {model_type} models.")
        return
    
    # Determine number of workers based on model type if not specified by user
    num_workers = args.workers if args.workers is not None else DEFAULT_WORKERS.get(model_type, 4)
    print(f"Processing with {num_workers} parallel workers")
    
    # Load conditions from the specified path
    conditions_path = "/home/zhang/metamat-agent/data/conditions.pt"
    if os.path.exists(conditions_path):
        print(f"Loading conditions from {conditions_path}")
        try:
            conditions = torch.load(conditions_path)
            print(f"Loaded {conditions.shape[0]} conditions, each with {conditions.shape[1]} parameters")
            # Debug: Print first few conditions
            for i in range(min(3, conditions.shape[0])):
                print(f"Sample condition {i}: {conditions[i]}")
            args.conditional = True
        except Exception as e:
            print(f"Error loading conditions: {e}")
            print("Falling back to unconditional generation")
            conditions = None
            args.conditional = False
    else:
        print(f"Warning: Conditions file {conditions_path} not found")
        print("Falling back to unconditional generation")
        conditions = None
        args.conditional = False
    
    # Initialize variables for structure generation
    generated_history = ""
    successful_structures = 0
    total_tokens = 0
    
    # Process structures in parallel 
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # Create a list of future tasks
        num_structures = min(args.num_structures, conditions.shape[0]) if conditions is not None else args.num_structures
        
        for i in range(num_structures):
            # Get condition if conditional generation is enabled
            condition = None
            if args.conditional and conditions is not None:
                condition = conditions[i].cpu().numpy().flatten()
                print(f"Structure {i}: Using condition {condition}")
            
            futures.append(
                executor.submit(
                    generate_structure,
                    i,
                    model_name,
                    model_type,
                    args.temperature,
                    args.max_tokens,
                    INIT_PROMPT,
                    generated_history,
                    condition
                )
            )
        
        # Process results as they complete with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating structures"):
            try:
                result = future.result()
                if result["success"]:
                    # Add to history to avoid duplicates
                    generated_history += "\n" + result["output"]
                    
                    # Save the structure
                    index = result["index"]
                    lattice_name = os.path.join(args.output_dir, f'{index}.npz')
                    
                    # Get the condition used for this structure
                    condition_used = None
                    if args.conditional and conditions is not None and index < conditions.shape[0]:
                        condition_used = conditions[index].cpu().numpy()
                        
                    # Save structure with condition information
                    np.savez(lattice_name,
                             atom_types=result["z"].cpu().numpy(),
                             lengths=result["lengths"].cpu().view(-1).numpy(),
                             angles=result["angles"].cpu().view(-1).numpy(),
                             frac_coords=result["coords"].cpu().numpy(),
                             edge_index=result["edge_index"].cpu().numpy(),
                             prop_list=condition_used,
                             )
                    
                    successful_structures += 1
                    total_tokens += result["token_count"]
                    print(f"Generated structure {index} with {result['num_atoms'][0]} nodes and {result['token_count']} tokens")
                    if condition_used is not None:
                        print(f"  Using condition: Young={condition_used[:3]}, Shear={condition_used[3:6]}, Poisson={condition_used[6:9]}")
                else:
                    print(f"Failed to generate structure {result['index']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error processing result: {e}")
    
    # Print summary
    print(f"\nGeneration complete!")
    print(f"Successfully generated {successful_structures}/{num_structures} structures")
    print(f"Total tokens: {total_tokens}")
    if successful_structures > 0:
        print(f"Average tokens per structure: {total_tokens / successful_structures:.2f}")
    
    # If using conditions, print summary of conditions used
    if args.conditional and conditions is not None:
        print(f"Used {num_structures} conditions from {conditions_path}")
    else:
        print("Generated structures without conditions")

if __name__ == "__main__":
    main()

