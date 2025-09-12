import os
import sys
import argparse
import torch
import re
import pandas as pd
import warnings
import concurrent.futures
from tqdm import tqdm

# Add path to allow imports from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Import the API functions from api.py
from api import generate_with_api

# Silence warnings
warnings.filterwarnings("ignore")

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

def parse_graph(text):
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
    # properties_start = text.index("Properties:")
    
    coord_block = text[coord_start:edge_start].strip().splitlines()
    edge_block  = text[edge_start + len("Edges:"):].strip().splitlines()
    # properties_block  = text[properties_start + len("Properties:"):].strip()
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
    lengths = torch.FloatTensor([[1,1,1]])
    angles = torch.FloatTensor([[90,90,90]])


    return z, coords, edge_index, batch, lengths, angles, num_atoms


# Baseline instruction prompt for metamaterial generation
INSTRUCTIONS_BASELINE_LLM="""
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

def parse_args():
    parser = argparse.ArgumentParser(description="Generate metamaterial structures using language models.")
    
    # Input and output arguments
    parser.add_argument("--input", type=str, default="data/metamaterial_design_prompts.csv",
                        help="Path to the input CSV file containing design prompts.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results.")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="llama-4-maverick",
                        help="Model to use for generation.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (higher values = more randomness).")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for processing prompts. Default is model-dependent.")
    
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
    
    # Evaluation options
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the generated structures.")
    parser.add_argument("--max_node_num", type=int, default=100,
                        help="Maximum number of nodes to generate.")
    
    return parser.parse_args()

def process_prompt(prompt, model_name, model_type, max_tokens, temperature):
    """Process a single prompt and return the result"""
    # Format conversation for API call
    conversation = [
        {"role": "system", "content": INSTRUCTIONS_BASELINE_LLM},
        {"role": "user", "content": prompt}
    ]
    
    # Get model response using the generate_with_api function
    output, token_count = generate_with_api(
        model_type=model_type,
        model=model_name,
        conversation=conversation,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return {
        "prompt": prompt,
        "output": output,
        "token_count": token_count
    }

def main():
    args = parse_args()
    
    # Determine model type using prefix-based logic
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
        print(f"Error: Unsupported model '{model_name}'. Cannot determine model type from name.")
        return
    
    # Update output directory to include model name
    args.output_dir = os.path.join("results", f"{model_name.replace('/', '-')}_prompt_guidance")
    
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
    
    # Load input prompts
    try:
        input_df = pd.read_csv(args.input, header=None)
        input_texts = input_df.values.tolist()
        print(f"Loaded {len(input_texts)} design prompts from {args.input}")
    except Exception as e:
        print(f"Error loading file {args.input}: {e}")
        return
    
    # Determine number of workers based on model type if not specified by user
    num_workers = args.workers if args.workers is not None else DEFAULT_WORKERS.get(model_type, 4)
    print(f"Processing prompts with {num_workers} parallel workers")
    
    # Initialize results container
    results = []

    # Process prompts in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of future tasks
        futures = []
        for input_row in input_texts:
            prompt = input_row[0]
            futures.append(
                executor.submit(
                    process_prompt,
                    prompt,
                    model_name,
                    model_type,
                    args.max_tokens,
                    args.temperature
                )
            )
        
        # Process results as they complete with tqdm for progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating structures"):
            try:
                result = future.result()
                results.append(result)
                print(f"Generated structure with {result['token_count']} tokens")
            except Exception as e:
                print(f"Error processing prompt: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame({
        'Prompt': [r['prompt'] for r in results],
        'Output': [r['output'] for r in results]
    })
    
    # Save results to CSV
    output_file = os.path.join(args.output_dir, f'results_{model_name.replace("/", "-")}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Statistics about generation
    total_tokens = sum(r['token_count'] for r in results)
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average tokens per prompt: {total_tokens / len(results) if results else 0:.2f}")
    
    # If evaluation is enabled, perform evaluation
    if args.evaluate:
        print("Evaluation functionality is not yet implemented.")
        # This would be where you'd implement the evaluation logic that was commented out
        # in the original code, using parse_graph_llm and MetamatGenAgent12

if __name__ == "__main__":
    main()


