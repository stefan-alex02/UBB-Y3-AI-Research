# Non-Nested Grid Search Explanation

Okay, you're observing the standard behavior of `GridSearchCV` (or `RandomizedSearchCV`) combined with Skorch's internal validation split, but let's clarify the flow because the logging can be a bit confusing.

**How `GridSearchCV` Works with the Data & Splits:**

1.  **Input:** `GridSearchCV` receives the *entire* `X_trainval` (347 samples) and `y_trainval`.
2.  **Outer CV (`cv=3`):** `GridSearchCV` uses the `cv_splitter = StratifiedKFold(n_splits=3)` to split `X_trainval` into 3 outer folds.
    *   Fold 1: ~2/3 for training (e.g., 231 samples), ~1/3 for testing *this parameter set* (e.g., 116 samples).
    *   Fold 2: Different ~2/3 for training (231 samples), different ~1/3 for testing (116 samples).
    *   Fold 3: Remaining ~2/3 for training (232 samples), remaining ~1/3 for testing (115 samples).
3.  **Parameter Candidates:** For *each* hyperparameter combination defined in `param_grid` (in your case, 2 candidates: `{lr=0.001, wd=0.01}` and `{lr=0.001, wd=0.005}`), `GridSearchCV` performs the following:
4.  **Fitting Each Outer Fold:** For a *single* parameter candidate (e.g., `{lr=0.001, wd=0.01}`):
    *   **Outer Fold 1:** It takes the ~231 training samples designated for this fold. It calls `estimator.fit(X_outer_train_1, y_outer_train_1)`.
        *   **Skorch Internal Split:** Inside `fit`, our `SkorchModelAdapter`'s `get_split_datasets` is called. `self.train_split` is `ValidSplit(cv=0.15, stratified=True)`. This splits the ~231 samples further.
            *   ~85% for *actual* training (~196 samples).
            *   ~15% for *internal* validation (~35 samples) used for `EarlyStopping` and `LRScheduler`.
            *   The adapter trains on the ~196, validates internally on the ~35. You see the epoch table for this inner fit (e.g., `Split created: 196 train, 35 validation.`).
    *   **Scoring Outer Fold 1:** After the inner fit completes (potentially stopping early), `GridSearchCV` takes the *best model from that inner fit* (restored by `EarlyStopping(load_best=True)`) and evaluates it on the ~116 *test samples* held out for Outer Fold 1 using the specified `scoring='accuracy'`. You see the debug logs `get_iterator creating PathImageDataset...` for this scoring step (size 116).
    *   **Outer Fold 2 & 3:** The process repeats for the other 2 outer folds using their respective train/test splits (e.g., fitting on ~231, testing on ~116; fitting on ~232, testing on ~115). Each fit has its own internal 85%/15% split.
5.  **Averaging Scores:** For the candidate `{lr=0.001, wd=0.01}`, `GridSearchCV` averages the scores obtained on the test sets of the 3 outer folds.
6.  **Repeat for Other Candidates:** Steps 4 & 5 are repeated for the *other* parameter candidate (`{lr=0.001, wd=0.005}`).
7.  **Select Best Parameters:** `GridSearchCV` compares the average scores for all candidates and identifies the best parameter set (e.g., `{lr=0.001, wd=0.01}` had an average score of `(0.552+0.664+0.548)/3 = 0.588`).
8.  **Refitting (`refit=True`):** This is the **crucial 7th fit** you observed.
    *   `GridSearchCV` takes the *best* parameter set it found (e.g., `{lr=0.001, wd=0.01}`).
    *   It calls `estimator.fit()` one last time, but this time using the **entire original `X_trainval` (all 347 samples)** and `y_trainval`.
    *   **Skorch Internal Split (Again):** Inside *this* final fit, `get_split_datasets` is called again. `ValidSplit(cv=0.15)` splits the full 347 samples into:
        *   ~85% for training (~294 samples).
        *   ~15% for internal validation (~53 samples).
        *   This is the log line: `Split created: 294 train, 53 validation.`
    *   The adapter trains on the ~294, validates on the ~53. You see the epoch table for this final refit.
    *   The model resulting from this final fit (using the best hyperparams and trained on almost all `trainval` data, potentially stopped early based on the internal 53 validation samples) is stored in `search.best_estimator_`.

**Summary & Addressing Your Concerns:**

*   **Data Usage:** During the 3-fold cross-validation *within* `GridSearchCV` (the first 6 fits), only a fraction (~2/3 * ~85% = ~56%) of the `trainval` data is used for *actual* gradient updates in each specific fit. The remaining data is used either for internal validation (~2/3 * ~15% = ~10%) or for scoring that fold (~1/3 = ~33%). This is standard CV practice to evaluate hyperparameters robustly.
*   **Purpose of the 7th Fit:** The final refit is essential. It trains a single model using the *best found hyperparameters* on *as much of the available training data (`X_trainval`) as possible*, while still using a small internal validation set for monitoring (like `EarlyStopping`). This refitted model (`search.best_estimator_`) is the one you typically want to use for final evaluation on the *real* unseen test set (`X_test`, `y_test`) or for deployment.
*   **Is Data Missing?** No data is truly missing. All `X_trainval` samples are used either for training, internal validation, or scoring *across the different folds and the final refit*. It's just that any *single* fit during the CV phase doesn't see all the data for training.

The process is working as intended by `GridSearchCV` with `refit=True` and Skorch's internal validation mechanism. The logs correctly reflect the different splits happening at different stages.



# Nested Grid Search Explanation

Okay, let's break down the nested CV logs and address your question about the data input for non-nested CV.

**Nested Cross-Validation (`nested_grid_search`) Log Analysis:**

1.  **Input Data:** `nested_grid_search` correctly receives the *full* dataset (434 samples in this case) because its purpose is to estimate the generalization performance of the *entire hyperparameter tuning process* across different data splits.
2.  **Outer Loop (`outer_cv=3`):** `cross_validate` splits the 434 samples into 3 outer folds.
    *   Fold 1: Train (~289), Test (~145)
    *   Fold 2: Train (~289), Test (~145)
    *   Fold 3: Train (~290), Test (~144)
3.  **Inner Loop (`inner_cv=2`, inside each outer fold):** For *each* outer fold, the `inner_search` (which is `GridSearchCV`) takes the *outer training set* (e.g., ~289 samples for Fold 1) and performs its own *inner* cross-validation to find the best hyperparameters *for that specific outer fold*.
    *   **GridSearchCV (`cv=2`):** Splits the ~289 samples into 2 inner folds.
        *   Inner Fold 1: Train (~145), Test (~144)
        *   Inner Fold 2: Train (~144), Test (~145)
    *   **Parameter Candidates:** For *each* parameter candidate (e.g., `{lr=0.001, wd=0.01}`):
        *   **Inner Fit 1 (on Inner Fold 1 Train):** Takes the ~145 samples. Calls `estimator.fit()`.
            *   **Innermost Skorch Split:** `ValidSplit(cv=0.15)` splits the ~145 samples into ~122 train / ~22 valid for `EarlyStopping`. Trains on ~122, validates on ~22. (Log: `Split created: 122 train, 22 validation.`)
        *   **Inner Score 1 (on Inner Fold 1 Test):** Evaluates the model from Inner Fit 1 on the ~144 inner test samples. (Log: `get_iterator creating PathImageDataset... size=144`)
        *   **Inner Fit 2 (on Inner Fold 2 Train):** Takes the ~144 samples. Calls `estimator.fit()`.
            *   **Innermost Skorch Split:** `ValidSplit(cv=0.15)` splits ~144 into ~123 train / ~22 valid. Trains on ~123, validates on ~22. (Log: `Split created: 123 train, 22 validation.`)
        *   **Inner Score 2 (on Inner Fold 2 Test):** Evaluates the model from Inner Fit 2 on the ~145 inner test samples. (Log: `get_iterator creating PathImageDataset... size=145`)
    *   **Candidate Score:** Averages the scores from Inner Score 1 and Inner Score 2 for this parameter candidate.
    *   **Repeat for other candidates:** The inner 2-fold CV (including the 85/15 splits) is repeated for the *other* parameter candidate (`{lr=0.001, wd=0.005}`).
    *   **Inner Best Params:** `GridSearchCV` selects the best parameters *for this outer fold*.
    *   **Inner Refit:** `GridSearchCV` refits a model using the best inner parameters on the *entire outer training set* (~289 samples). This again involves the 85/15 Skorch split (~245/~44). (Log: `Split created: 245 train, 44 validation.`)
4.  **Outer Score:** `cross_validate` takes the refitted model from the inner loop and evaluates it on the *outer test set* (~145 samples) held out for this outer fold. (Log: `get_iterator creating PathImageDataset... size=145`). This gives the final score for the outer fold (e.g., `accuracy: (test=0.648)`).
5.  **Repeat Outer Folds:** Steps 3 and 4 are repeated for the other 2 outer folds.
6.  **Final Result:** The scores from the outer test sets are averaged to give the final nested CV performance estimate (e.g., `Mean Test Accuracy: 0.6360 +/- 0.0157`).

**Data Usage in Nested CV:** Correct. It uses the *full dataset*, splitting it multiple times to get an unbiased estimate. No data is truly "missing" overall, but each individual fit sees only a fraction.

---

## **Data Input for Non-Nested Grid Search (`non_nested_grid_search`)**

Your question is crucial: should non-nested search use the *full* dataset or only the `trainval` portion?

*   **Purpose:** The goal of non-nested grid search is typically to find the best hyperparameters using the available *training* data and then evaluate the *single best model* (refit on all training data) on a *separate, held-out test set*.
*   **Data Leakage:** If you include the final `test` data (`X_test`, `y_test` defined by `ImageDatasetHandler` for FLAT datasets or the fixed `test` folder) within the data passed to `GridSearchCV` (e.g., `X_full`), then the cross-validation folds *within* `GridSearchCV` will inevitably use parts of your intended final test set for training or validation during the hyperparameter search. This leads to **data leakage**. The hyperparameter selection process gets information from the test set, making the final evaluation on that same test set overly optimistic and not a true reflection of generalization performance.
*   **Correct Approach:** You should **only pass the `trainval` data** (`X_trainval`, `y_trainval` obtained from `self.dataset_handler.get_train_val_paths_labels()`) to `GridSearchCV` in the `non_nested_grid_search` method. The `X_test`, `y_test` should be kept completely separate and used *only* for the final evaluation step performed by the *next* method in the pipeline (e.g., `single_eval`).

**Your current `non_nested_grid_search` implementation correctly uses only `X_trainval`, `y_trainval` for the `search.fit()` call.** This prevents data leakage and aligns with the standard practice for non-nested hyperparameter tuning followed by independent testing.

**Summary:**

*   **Nested CV:** Correctly uses the **full dataset** (`X_full`, `y_full`) because it estimates the performance of the *entire tuning process*. Test data is used for *outer* fold evaluation only.
*   **Non-Nested CV:** Correctly uses only the **training+validation data** (`X_trainval`, `y_trainval`) for the `GridSearchCV` fitting process. The separate `test` data should only be touched *after* the grid search is complete, typically by a subsequent `single_eval` step using the refitted `best_estimator_`.