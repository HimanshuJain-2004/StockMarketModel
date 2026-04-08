import keras_tuner as kt
from keras_tuner.oracles import BayesianOptimizationOracle
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import r2_score
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress INFO + WARNING
tf.keras.utils.disable_interactive_logging()
tf.get_logger().setLevel('ERROR')
# -----------------------------
# GLOBAL SEED (MATCH MAIN FILE)
# -----------------------------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

# Custom tuner for non-Keras training loops
class REGCNTuner(kt.Tuner):
    def save_model(self, trial_id, model, step=0):
            return
    def run_trial(self, trial, data, s_index, seq_len):
        # FIX: disable save_model() to avoid NotImplementedError
        
        hp = trial.hyperparameters

        # hyperparameters to tune
        lr = hp.Choice("lr", [1e-4, 5e-4, 8e-4])
        n_neurons = hp.Choice("n_neurons", [64, 128, 192])
        r_acc = hp.Choice("r_acc", [0.05, 0.1, 0.15])

        # import here to avoid circular import issues
        from Proposed_model import main

        # run your custom GCGRU REGCN model
        pred, y_test = main(
            data=data,
            s_index=s_index,
            lr=lr,
            n_neurons=n_neurons,
            seq_len=seq_len,
            n_epochs=30,          # smaller epochs for tuning
            r_acc=r_acc,
            tune_mode=True
        )

        y_true = y_test.reshape(-1)
        y_pred = pred.reshape(-1)

        # compute R2
        r2 = r2_score(y_true, y_pred)

        # tuner minimizes → use negative R2
        score = -r2

        # Report the score back to Keras Tuner
        self.oracle.update_trial(
            trial.trial_id,
            {"score": score}
        )


def run_tuner(data, s_index, seq_len):

    oracle = BayesianOptimizationOracle(
        objective=kt.Objective("score", direction="min"),
        max_trials=10
    )

    tuner = REGCNTuner(
        oracle=oracle,
        directory="/kaggle/working/tuner",
        project_name=f"regcn_stock_{s_index}"
    )

    print(f"\n🔍 Running Tuning for Stock {s_index} ...")

    # Start tuning
    tuner.search(
        data=data,
        s_index=s_index,
        seq_len=seq_len
    )

    # Extract best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]

    best_params = {
        "lr": best_hp.get("lr"),
        "n_neurons": best_hp.get("n_neurons"),
        "r_acc": best_hp.get("r_acc")
    }

    print(f"\n🔥 Best Hyperparameters for Stock {s_index}:")
    print(best_params)

    return best_params
