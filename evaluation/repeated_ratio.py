import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from model.modal_agent12 import parse_graph_llm
import pandas as pd
from tqdm import tqdm

from scipy.spatial.distance import cdist


def compute_repeated_ratio(coords, edge_indexs, lengths, angles, dist_threshold=.1):
    """
    Compute the repeated ratio of a graph.

    Args:
        coords (torch.Tensor): Node coordinates.
        edge_index (torch.Tensor): Edge indices.
        batch (torch.Tensor): Batch indices.

    Returns:
        float: Repeated ratio.
    """
    # Implement the logic to compute the repeated ratio
    # This is a placeholder implementation
    repeated_num = 0
    for i in tqdm(range(len(coords))):
        for j in range(i + 1, len(coords)):
            if coords[i].shape != coords[j].shape:
                continue
            # Calculate the distance between the two nodes
            dist = cdist(coords[i], coords[j], 'euclidean')
            max_pair_dist = dist.min(axis=1).max()
            # Check if the distance is less than a threshold (e.g., 1.0)
            if max_pair_dist < dist_threshold:
                repeated_num += 1
                break

    repeated_ratio = repeated_num / len(coords)
    print('repeated_num', repeated_num)
    print('len(coords)', len(coords))
    return repeated_ratio




# results_df = pd.read_csv( 'D:\ModalAgent\evaluation\qwen3-235b-a22b_prompt_guidance\\results_qwen3-235b-a22b.csv')
# results_df = pd.read_csv( 'D:\ModalAgent\evaluation\deepseek-chat_prompt_guidance\\results_deepseek-chat.csv')
# results_df = pd.read_csv( 'D:\ModalAgent\evaluation\gemini-2.0-flash-lite_prompt_guidance\\results_0.4755.csv')
# results_df = pd.read_csv( 'D:\ModalAgent\evaluation\qwen3-235b-a22b_prompt_guidance\\results_qwen3-235b-a22b.csv')
# results_df = pd.read_csv( 'D:\ModalAgent\evaluation\qwen3-235b-a22b_prompt_guidance\\results_qwen3-235b-a22b.csv')


output_text = results_df['Output'].values.tolist()
all_coords = []
all_edge_indexs = []
all_lengths = []
all_angles = []
for i, output in enumerate(tqdm(output_text)):
    input_text = results_df['Prompt'].values.tolist()[i]
    # print('output', output)
    # Parse the graph from the output
    # z, coords, edge_index, batch, lengths, angles, num_atoms = parse_graph(output)
    z, coords, edge_index, batch, lengths, angles, num_atoms = parse_graph_llm(output)
    # visualizeLattice(coords.cpu().numpy(), edge_index.cpu().numpy())
    all_coords.append(coords)
    all_edge_indexs.append(edge_index)
    all_lengths.append(lengths)
    all_angles.append(angles)

    # Save the results to a CSV file
repeat_ratio = compute_repeated_ratio(all_coords, all_edge_indexs, all_lengths, all_angles, dist_threshold=0.05)
print(f"Repeated Ratio: {repeat_ratio:.4f}")

