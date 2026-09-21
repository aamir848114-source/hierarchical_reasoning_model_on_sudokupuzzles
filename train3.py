import os
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from metrics import tdoku_backtracking_steps
from recurrrenttrasnfroemrtest import RecurrentTransformer, TransformerConfig
from HRM_TEST1 import HierarchicalReasoningModelCore, HRMConfig
from SATNet_test1 import SudokuSATNet, SATNetConfig
from rnn_test import RecurrentRelationalNetwork, RRNConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

class SudokuDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_excel(file_path)

        self.puzzles = df["question"].astype(str).values
        self.solutions = df["answer"].astype(str).values

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = [0 if c in [".", "0"] else int(c) for c in self.puzzles[idx]]
        solution = [0 if c in [".", "0"] else int(c) for c in self.solutions[idx]]
        return (
            torch.tensor(puzzle, dtype=torch.long).reshape(9, 9),
            torch.tensor(solution, dtype=torch.long).reshape(9, 9),
        )


def load_sudoku_data(file_path, batch_size=32, shuffle=True):
    dataset = SudokuDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

IGNORE_INDEX = -100

def map_targets_for_loss(solutions):

    targets = solutions.view(solutions.size(0), -1)
    return torch.where(targets > 0, targets - 1, torch.full_like(targets, IGNORE_INDEX))


def prepare_model_logits(predictions, batch):

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.dim() == 3 and predictions.size(1) == 81 and predictions.size(2) in (9, 10):
        logits = predictions
    elif predictions.dim() == 2:
        last = predictions.size(-1)
        if last == 81 * 9:
            logits = predictions.view(batch, 81, 9)
        elif last == 81 * 10:
            logits = predictions.view(batch, 81, 10)
        elif last == 729:
            logits = predictions.view(batch, 81, 9)
        elif last == 810:
            logits = predictions.view(batch, 81, 10)
        else:
            logits = predictions.view(batch, 81, -1)
    else:
        logits = predictions.view(batch, 81, -1)

    if logits.size(-1) == 10:
        logits = logits[..., 1:]

    return logits

def create_sudoku_adjacency_matrix():
    adj = torch.zeros(81, 81)
    for i in range(81):
        r, c = divmod(i, 9)
        for j in range(81):
            r2, c2 = divmod(j, 9)
            if r == r2 or c == c2 or (r // 3 == r2 // 3 and c // 3 == c2 // 3):
                adj[i, j] = 1
    return adj.unsqueeze(0).unsqueeze(-1)

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """
    Augment a Sudoku puzzle and solution by:
    - Random digit permutation (1..9, with 0 unchanged)
    - Random transpose
    - Random row and column permutations (respecting bands/stacks)
    """
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)
    def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
        """
        Digit Permutation: Random 1-9 mapping (0/blank unchanged)
        Transpose: 50% probability of swapping rows ↔ columns
        Band/Stack Shuffling: Permutes 3×3 regions while respecting Sudoku structure
        """
        import numpy as np
        board = board.copy()
        solution = solution.copy()

        # Digit permutation
        digits = np.arange(1, 10)
        perm = np.random.permutation(digits)
        digit_map = {d: p for d, p in zip(digits, perm)}
        def permute_digits(arr):
            arr_perm = arr.copy()
            for d in digits:
                arr_perm[arr == d] = digit_map[d]
            return arr_perm
        board = permute_digits(board)
        solution = permute_digits(solution)

        # Transpose with 50% probability
        if np.random.rand() < 0.5:
            board = board.T
            solution = solution.T

        # Band shuffling (shuffle row blocks)
        band_indices = np.arange(3)
        np.random.shuffle(band_indices)
        new_board = np.zeros_like(board)
        new_solution = np.zeros_like(solution)
        for i, bi in enumerate(band_indices):
            new_board[i*3:(i+1)*3, :] = board[bi*3:(bi+1)*3, :]
            new_solution[i*3:(i+1)*3, :] = solution[bi*3:(bi+1)*3, :]
        board = new_board
        solution = new_solution

        # Stack shuffling (shuffle column blocks)
        stack_indices = np.arange(3)
        np.random.shuffle(stack_indices)
        new_board = np.zeros_like(board)
        new_solution = np.zeros_like(solution)
        for i, si in enumerate(stack_indices):
            new_board[:, i*3:(i+1)*3] = board[:, si*3:(si+1)*3]
            new_solution[:, i*3:(i+1)*3] = solution[:, si*3:(si+1)*3]
        board = new_board
        solution = new_solution

        return board, solution


def generate_augmented_dataset(input_file: str, output_file: str, num_augments: int = 1000):
    """
    Generate augmented Sudoku dataset by creating num_augments per puzzle.
    Saves to output_file as Excel.
    """
    print(f"Loading {input_file}...")
    df = pd.read_excel(input_file)
    
    puzzles = df["question"].astype(str).values
    solutions = df["answer"].astype(str).values
    
    augmented_puzzles = []
    augmented_solutions = []
    
    print(f"Generating {num_augments} augmentations per puzzle...")
    for idx, (puzzle_str, solution_str) in enumerate(tqdm(zip(puzzles, solutions))):
        # Parse original puzzle and solution
        puzzle = np.array([0 if c in [".", "0"] else int(c) for c in puzzle_str]).reshape(9, 9)
        solution = np.array([0 if c in [".", "0"] else int(c) for c in solution_str]).reshape(9, 9)
        
        # Add original (no augmentation)
        augmented_puzzles.append(puzzle_str)
        augmented_solutions.append(solution_str)
        
        # Generate augmentations
        for aug_idx in range(num_augments):
            aug_puzzle, aug_solution = shuffle_sudoku(puzzle, solution)
            
            # Convert back to string format
            puzzle_str_aug = "".join([str(c) if c > 0 else "." for c in aug_puzzle.flatten()])
            solution_str_aug = "".join([str(c) if c > 0 else "." for c in aug_solution.flatten()])
            
            augmented_puzzles.append(puzzle_str_aug)
            augmented_solutions.append(solution_str_aug)
    
    # Save augmented dataset to Excel
    print(f"Saving augmented dataset to {output_file}...")
    augmented_df = pd.DataFrame({
        "question": augmented_puzzles,
        "answer": augmented_solutions
    })
    augmented_df.to_excel(output_file, index=False)
    print(f"Augmented dataset saved: {output_file} with {len(augmented_df)} examples")


def train_model(model, train_loader, test_loader, model_name, epochs=10, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    results = {"training_loss": [], "test_accuracy": 0.0, "backtracking_count": 0}
    print(f"\n=== Training {model_name.upper()} ===")

    # Track backtracking
    total_backtracking = 0


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for puzzles, solutions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            puzzles, solutions = puzzles.to(device), solutions.to(device)
            batch = puzzles.size(0)

            optimizer.zero_grad()

            # ...existing code...
            if model_name == "satnet":
                puzzles_flat = puzzles.view(batch, -1)
                x = torch.stack([(puzzles_flat == d).float() for d in range(1, 10)], dim=-1)
                x = x.view(batch, -1)  # (batch,729)
                predictions = model(x)
                logits = prepare_model_logits(predictions, batch)
                targets = map_targets_for_loss(solutions)
                loss = criterion(logits.view(-1, 9), targets.view(-1))
            elif model_name == "rrn":
                x = puzzles.view(batch, -1, 1).float()
                edge_index = create_sudoku_adjacency_matrix().to(device)
                edge_index = edge_index.squeeze(0)
                edge_index = edge_index.unsqueeze(0).repeat(batch, 1, 1, 1)
                predictions = model(x, edge_index)
                logits = prepare_model_logits(predictions, batch)
                targets = map_targets_for_loss(solutions)
                loss = criterion(logits.view(-1, 9), targets.view(-1))
            elif model_name == "hrm":
                predictions, q_values, extras = model(puzzles.view(batch, -1), return_intermediate=True)
                logits = prepare_model_logits(predictions, batch)
                targets = map_targets_for_loss(solutions)
                loss = criterion(logits.view(-1, 9), targets.view(-1))
                if extras.get("intermediate_outputs"):
                    ds_loss = 0
                    for int_output in extras["intermediate_outputs"]:
                        int_logits = prepare_model_logits(int_output, batch)
                        ds_loss += criterion(int_logits.view(-1, 9), targets.view(-1))
                    loss = loss + 0.1 * ds_loss
                if extras.get("remainders"):
                    ponder_loss = sum(extras["remainders"]).mean()
                    loss = loss + 0.01 * ponder_loss
            else:
                predictions = model(puzzles.view(batch, -1))
                logits = prepare_model_logits(predictions, batch)
                targets = map_targets_for_loss(solutions)
                loss = criterion(logits.view(-1, 9), targets.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Track backtracking if available in model/extras
            backtracking = 0
            if model_name == "hrm":
                if 'extras' in locals() and extras is not None:
                    backtracking = extras.get("backtracking_count", 0)
            else:
                if hasattr(model, "backtracking_count"):
                    backtracking = getattr(model, "backtracking_count", 0)
            total_backtracking += backtracking

        epoch_loss = total_loss / max(1, len(train_loader))
        results["training_loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    intermediate_timesteps = []

    # For backtracking in test
    test_backtracking = 0

    with torch.no_grad():
        for puzzles, solutions in tqdm(test_loader, desc="Testing"):
            puzzles, solutions = puzzles.to(device), solutions.to(device)
            batch = puzzles.size(0)

            # ...existing code...
            if model_name == "satnet":
                puzzles_flat = puzzles.view(batch, -1)
                x = torch.stack([(puzzles_flat == d).float() for d in range(1, 10)], dim=-1).view(batch, -1)
                predictions = model(x)
                logits = prepare_model_logits(predictions, batch)
                if hasattr(model, "intermediate_outputs"):
                    intermediate_timesteps.append(model.intermediate_outputs)
            elif model_name == "rrn":
                x = puzzles.view(batch, -1, 1).float()
                edge_index = create_sudoku_adjacency_matrix().to(device)
                edge_index = edge_index.squeeze(0)
                edge_index = edge_index.unsqueeze(0).repeat(batch, 1, 1, 1)
                predictions = model(x, edge_index)
                logits = prepare_model_logits(predictions, batch)
                if hasattr(model, "intermediate_outputs"):
                    intermediate_timesteps.append(model.intermediate_outputs)
            elif model_name == "transformer":
                if hasattr(model, "get_intermediate_outputs"):
                    predictions, intermediate_outputs = model(puzzles.view(batch, -1), return_intermediate=True)
                    logits = prepare_model_logits(predictions, batch)
                    intermediate_timesteps.append(intermediate_outputs)
                else:
                    predictions = model(puzzles.view(batch, -1))
                    logits = prepare_model_logits(predictions, batch)
            elif model_name == "hrm":
                predictions, q_values, extras = model(puzzles.view(batch, -1), return_intermediate=True)
                logits = prepare_model_logits(predictions, batch)
                # Backtracking from extras
                test_backtracking += extras.get("backtracking_count", 0)
            else:
                predictions = model(puzzles.view(batch, -1))
                logits = prepare_model_logits(predictions, batch)
            # Backtracking from model attributes if available
            if hasattr(model, "backtracking_count"):
                test_backtracking += getattr(model, "backtracking_count", 0)

            preds = logits.argmax(dim=-1) + 1
            preds_flat = preds.view(-1)
            solutions_flat = solutions.view(-1)

            mask = solutions_flat > 0
            correct += ((preds_flat == solutions_flat) & mask).sum().item()
            total += mask.sum().item()

            all_preds.extend(preds_flat[mask].cpu().numpy().tolist())
            all_labels.extend(solutions_flat[mask].cpu().numpy().tolist())

            # Tdoku backtracking metric for each predicted board in batch
            for i in range(batch):
                pred_board = preds[i].cpu().numpy().reshape(9, 9)
                # Convert blanks (values < 1 or > 9) to 0
                pred_board = np.where((pred_board < 1) | (pred_board > 9), 0, pred_board)
                steps = tdoku_backtracking_steps(pred_board)
                results.setdefault("tdoku_backtracking_steps", []).append(steps)
    acc = 100.0 * correct / total if total > 0 else 0.0
    results["test_accuracy"] = acc

    # Final backtracking metric
    # Average backtracking count across all test batches
    avg_backtracking = test_backtracking / max(1, len(test_loader)) if test_backtracking > 0 else (total_backtracking / max(1, len(train_loader)) if total_backtracking > 0 else 0)
    results["backtracking_count"] = avg_backtracking

    # Report tdoku backtracking metric
    if "tdoku_backtracking_steps" in results and results["tdoku_backtracking_steps"]:
        avg_tdoku_steps = np.mean(results["tdoku_backtracking_steps"])
        print(f"Average tdoku backtracking steps for {model_name.upper()}: {avg_tdoku_steps:.2f}")
        results["avg_tdoku_backtracking_steps"] = avg_tdoku_steps

    print(f"Test Accuracy for {model_name.upper()}: {acc:.2f}% (evaluated on non-blank cells only)")
    print(f"Average Backtracking Count for {model_name.upper()}: {avg_backtracking:.2f}")

    if len(all_labels) > 0:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(1, 10)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=range(1, 10), yticklabels=range(1, 10))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"Confusion Matrix: {model_name}")
        plt.savefig(f"results/{model_name}_confusion_matrix.png")
        plt.close()

    # Save visualizations of model timesteps for all models
    def plot_sudoku_timesteps(initial, timesteps, model_name):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        n_steps = len(timesteps)
        fig, axes = plt.subplots(1, n_steps + 2, figsize=(2.5 * (n_steps + 2), 4))
        # Initial board
        axes[0].set_title("Initial")
        for i in range(9):
            for j in range(9):
                val = initial[i, j]
                if val > 0:
                    axes[0].text(j, i, str(val), va='center', ha='center', fontsize=14, fontweight='bold', color='black')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlim(-0.5, 8.5)
        axes[0].set_ylim(-0.5, 8.5)
        # Timesteps
        initial_mask = (initial > 0)
        prev = initial.copy()
        for t, board in enumerate(timesteps):
            axes[t+1].set_title(f"Timestep i = {t}")
            for i in range(9):
                for j in range(9):
                    val = board[i, j]
                    if initial_mask[i, j]:
                        axes[t+1].text(j, i, str(val), va='center', ha='center', fontsize=14, fontweight='bold', color='black')
                    elif val > 0 and prev[i, j] != val:
                        axes[t+1].text(j, i, str(val), va='center', ha='center', fontsize=14, color='red')
                    elif val > 0:
                        axes[t+1].text(j, i, str(val), va='center', ha='center', fontsize=14, color='grey')
            axes[t+1].set_xticks([])
            axes[t+1].set_yticks([])
            axes[t+1].set_xlim(-0.5, 8.5)
            axes[t+1].set_ylim(-0.5, 8.5)
            prev = board.copy()
        # Final board
        axes[-1].set_title("Final")
        final = timesteps[-1]
        for i in range(9):
            for j in range(9):
                val = final[i, j]
                if initial_mask[i, j]:
                    axes[-1].text(j, i, str(val), va='center', ha='center', fontsize=14, fontweight='bold', color='black')
                elif val > 0:
                    axes[-1].text(j, i, str(val), va='center', ha='center', fontsize=14, color='red')
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
        axes[-1].set_xlim(-0.5, 8.5)
        axes[-1].set_ylim(-0.5, 8.5)
        # Legend
        black_patch = mpatches.Patch(color='black', label='Initial filled cells')
        red_patch = mpatches.Patch(color='red', label='Newly filled cells')
        grey_patch = mpatches.Patch(color='grey', label='Changed cells from previous timestep')
        plt.legend(handles=[black_patch, red_patch, grey_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_timesteps.png")
        plt.close()

    # Collect timesteps for each model (example: use first batch in test_loader)
    try:
        for puzzles, solutions in test_loader:
            initial = puzzles[0].cpu().numpy()
            puzzles = puzzles.to(device)
            batch = puzzles.size(0)

            # For HRM, collect intermediate outputs
            if model_name == "hrm":
                predictions, q_values, extras = model(puzzles.view(batch, -1), return_intermediate=True)
                timesteps = []
                for int_output in extras["intermediate_outputs"]:
                    pred = int_output[0].argmax(dim=-1).cpu().numpy().reshape(9, 9)
                    timesteps.append(pred)
                plot_sudoku_timesteps(initial, timesteps, model_name)
            # For SATNet: collect intermediate outputs
            elif model_name == "satnet":
                puzzles_flat = puzzles.view(batch, -1)
                x = torch.stack([(puzzles_flat == d).float() for d in range(1, 10)], dim=-1).view(batch, -1)
                predictions, layerwise_outputs = model(x, return_intermediate=True)
                timesteps = [out[0].argmax(dim=-1).cpu().numpy().reshape(9, 9) for out in layerwise_outputs]
                plot_sudoku_timesteps(initial, timesteps, model_name)
            # For RRN: collect intermediate outputs
            elif model_name == "rrn":
                x = puzzles.view(batch, -1, 1).float()
                edge_index = create_sudoku_adjacency_matrix().to(device)
                edge_index = edge_index.squeeze(0)
                edge_index = edge_index.unsqueeze(0).repeat(batch, 1, 1, 1)
                predictions, layerwise_outputs = model(x, edge_index, return_intermediate=True)
                timesteps = [out[0].argmax(dim=-1).cpu().numpy().reshape(9, 9) for out in layerwise_outputs]
                plot_sudoku_timesteps(initial, timesteps, model_name)
            # For Transformer: collect intermediate outputs
            elif model_name == "transformer":
                predictions, layerwise_outputs = model(puzzles.view(batch, -1).long(), return_intermediate=True)
                timesteps = [out[0].argmax(dim=-1).cpu().numpy().reshape(9, 9) for out in layerwise_outputs]
                plot_sudoku_timesteps(initial, timesteps, model_name)
            else:
                predictions = model(puzzles.view(batch, -1))
                final_pred = predictions[0].argmax(dim=-1).cpu().numpy().reshape(9, 9)
                plot_sudoku_timesteps(initial, [final_pred], model_name)
            break
    except Exception as e:
        print(f"Warning: Could not generate timestep visualization for {model_name}: {e}")
    
    # Save model weights
    try:
        model_save_path = f"results/{model_name}_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved: {model_save_path}")
    except Exception as e:
        print(f"Warning: Could not save model {model_name}: {e}")
    
    return results

def save_results_to_file(all_results, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = os.path.join("results", f"final_results_{timestamp}.txt")
    
    with open(results_path, "w") as f:
        f.write("=== FINAL RESULTS ===\n")
        for name, res in all_results.items():
            last_loss = res["training_loss"][-1] if res["training_loss"] else float("nan")
            f.write(f"{name.upper()}: Test Accuracy = {res['test_accuracy']:.2f}%, Final Loss = {last_loss:.4f}\n")
            f.write(f"{name.upper()}: Backtracking Count = {res.get('backtracking_count', 0)}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="all", choices=["all", "transformer", "hrm", "satnet", "rrn"]
    )
    parser.add_argument("--train_data", default="train.xlsx")
    parser.add_argument("--test_data", default="test_hard.xlsx")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_augments", type=int, default=1000, help="Number of augmentations per puzzle")
    parser.add_argument("--use_original", type=str, default=None, choices=["y", "n"], help="Use only original 1000 examples (y/n)")
    args = parser.parse_args()

    # Ask user if they want to use original 1000 examples or full augmented dataset
    use_original = args.use_original
    if use_original is None:
        use_original = input("Use only original 1000 examples for training? (y/n): ").strip().lower()
    
    if use_original == "y":
        print("\nUsing original 1000 examples for training...")
        train_file = args.train_data
    else:
        # Generate augmented dataset if it doesn't exist
        augmented_train_file = "train_aug.xlsx"
        if not os.path.exists(augmented_train_file):
            print(f"\n{augmented_train_file} not found. Generating augmented dataset...")
            generate_augmented_dataset(args.train_data, augmented_train_file, num_augments=args.num_augments)
        else:
            print(f"\n{augmented_train_file} already exists. Skipping generation.")
        train_file = augmented_train_file

    # Ask user if they want to train only HRM or all models
    train_hrm_only = args.model == "hrm"
    if args.model == "all":
        train_hrm_only = input("Train only HRM? (y/n): ").strip().lower() == "y"

    if train_hrm_only:
        print("\nTraining only HRM model...")
        args.model = "hrm"
    else:
        print("\nTraining all models including HRM...")
        args.model = "all"

    # Load data
    train_loader = load_sudoku_data(train_file, batch_size=args.batch_size)
    test_loader = load_sudoku_data(args.test_data, batch_size=args.batch_size)

    models = []

    if args.model in ["all", "transformer"] and globals().get("RecurrentTransformer") is not None:
        models.append(("transformer", RecurrentTransformer(TransformerConfig()).to(device)))

    if args.model in ["all", "hrm"] and globals().get("HierarchicalReasoningModelCore") is not None:
        models.append(("hrm", HierarchicalReasoningModelCore(HRMConfig()).to(device)))

    if args.model in ["all", "satnet"] and globals().get("SudokuSATNet") is not None:
        satnet_config = SATNetConfig(n_vars=729, n_constraints=324, hidden_dim=512)
        models.append(("satnet", SudokuSATNet(satnet_config).to(device)))

    if args.model in ["all", "rrn"] and globals().get("RecurrentRelationalNetwork") is not None:
        models.append(
            (
                "rrn",
                RecurrentRelationalNetwork(
                    RRNConfig(
                        input_dim=1,
                        hidden_dim=128,
                        message_dim=128,
                        edge_dim=1,
                        output_dim=10,
                    )
                ).to(device),
            )
        )

    if not models:
        print("No models to train (check imports / --model argument). Exiting.")
        return

    print("Models to train:", [name for name, _ in models])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {}
    for name, model in models:
        lr = 1e-4 if name == "satnet" else args.lr
        all_results[name] = train_model(model, train_loader, test_loader, name, args.epochs, lr)

    print("\n=== FINAL RESULTS ===")
    for name, res in all_results.items():
        last_loss = res["training_loss"][-1] if res["training_loss"] else float("nan")
        print(f"{name.upper()}: Test Accuracy = {res['test_accuracy']:.2f}%, Final Loss = {last_loss:.4f}")

    save_results_to_file(all_results, timestamp)
    print(f"\nResults saved to: results/final_results_{timestamp}.txt")


if __name__ == "__main__":
    main()