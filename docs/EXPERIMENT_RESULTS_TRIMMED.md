# Experiment Results
**Note:** Showing top 5 models per subset (by avg MAE across horizons). Baseline always included.


## Initial

### METR-LA

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_BS200_XAVIER`: Baseline with batch_size=200, 20 epochs, early_stop=5, xavier init
- `STGFORMER_DOW`: Baseline with batch_size=200, 20 epochs, early_stop=5 and DOW embeddings
- `STGFORMER_BS200_EXCLUDE_MISSING`: Baseline with batch_size=200, exclude missing values from normalization

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_BS200_XAVIER   | STGFORMER_DOW   | STGFORMER_BS200_EXCLUDE_MISSING   |
|:-------|:---------|:------------------------|:-------------------------|:----------------|:----------------------------------|
| 15 min | MAE      | 2.616                   | 2.619                    | **2.607**       | 2.608                             |
|        | RMSE     | 4.944                   | 4.962                    | 4.953           | **4.929**                         |
|        | MAPE     | 6.636%                  | 6.634%                   | 6.663%          | **6.630%**                        |
|        |          |                         |                          |                 |                                   |
| 30 min | MAE      | 2.862                   | 2.872                    | **2.843**       | 2.858                             |
|        | RMSE     | **5.630**               | 5.671                    | 5.641           | 5.637                             |
|        | MAPE     | **7.533%**              | 7.565%                   | 7.579%          | 7.549%                            |
|        |          |                         |                          |                 |                                   |
| 1 hour | MAE      | 3.167                   | 3.193                    | **3.127**       | 3.165                             |
|        | RMSE     | 6.440                   | 6.511                    | **6.417**       | 6.458                             |
|        | MAPE     | **8.683%**              | 8.764%                   | 8.705%          | 8.730%                            |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_BS200_XAVIER`: Baseline with batch_size=200, 20 epochs, early_stop=5, xavier init
- `STGFORMER_DOW`: Baseline with batch_size=200, 20 epochs, early_stop=5 and DOW embeddings
- `STGFORMER_BS200_EXCLUDE_MISSING`: Baseline with batch_size=200, exclude missing values from normalization

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_BS200_XAVIER   | STGFORMER_DOW   | STGFORMER_BS200_EXCLUDE_MISSING   |
|:-------|:---------|:------------------------|:-------------------------|:----------------|:----------------------------------|
| 15 min | MAE      | 1.138                   | 1.137                    | 1.135           | **1.134**                         |
|        | RMSE     | 2.349                   | 2.326                    | 2.332           | **2.317**                         |
|        | MAPE     | 2.372%                  | 2.363%                   | 2.364%          | **2.315%**                        |
|        |          |                         |                          |                 |                                   |
| 30 min | MAE      | 1.356                   | 1.353                    | **1.347**       | 1.352                             |
|        | RMSE     | 3.001                   | 2.975                    | 2.972           | **2.968**                         |
|        | MAPE     | 2.934%                  | 2.924%                   | 2.921%          | **2.877%**                        |
|        |          |                         |                          |                 |                                   |
| 1 hour | MAE      | 1.599                   | 1.595                    | **1.581**       | 1.594                             |
|        | RMSE     | 3.677                   | 3.648                    | **3.623**       | 3.637                             |
|        | MAPE     | 3.594%                  | 3.591%                   | 3.572%          | **3.560%**                        |

## Spatial

### METR-LA

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_GEO_CHEB`: Geographic graph + Chebyshev polynomial propagation
- `STGFORMER_HYBRID`: Hybrid graph (geographic + learned)
- `STGFORMER_CHEBYSHEV`: Chebyshev polynomial propagation
- `STGFORMER_HYBRID_CHEB`: Hybrid graph + Chebyshev polynomial propagation

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_GEO_CHEB   | STGFORMER_HYBRID   | STGFORMER_CHEBYSHEV   | STGFORMER_HYBRID_CHEB   |
|:-------|:---------|:------------------------|:---------------------|:-------------------|:----------------------|:------------------------|
| 15 min | MAE      | 2.616                   | **2.592**            | 2.595              | 2.594                 | 2.597                   |
|        | RMSE     | 4.944                   | 4.921                | 4.936              | **4.898**             | 4.898                   |
|        | MAPE     | **6.636%**              | 6.681%               | 6.729%             | 6.684%                | 6.677%                  |
|        |          |                         |                      |                    |                       |                         |
| 30 min | MAE      | 2.862                   | **2.835**            | 2.837              | 2.836                 | 2.839                   |
|        | RMSE     | 5.630                   | 5.633                | 5.644              | 5.609                 | **5.604**               |
|        | MAPE     | **7.533%**              | 7.584%               | 7.686%             | 7.575%                | 7.540%                  |
|        |          |                         |                      |                    |                       |                         |
| 1 hour | MAE      | 3.167                   | 3.136                | **3.136**          | 3.140                 | 3.143                   |
|        | RMSE     | 6.440                   | 6.432                | 6.455              | 6.430                 | **6.419**               |
|        | MAPE     | 8.683%                  | 8.734%               | 8.868%             | 8.713%                | **8.656%**              |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_HYBRID_CHEB`: Hybrid graph + Chebyshev polynomial propagation
- `STGFORMER_SPECTRAL_INIT`: Learned graph initialized from Laplacian eigenvectors
- `STGFORMER_CHEBYSHEV`: Chebyshev polynomial propagation
- `STGFORMER_GEO`: Geographic (pre-computed) graph

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_HYBRID_CHEB   | STGFORMER_SPECTRAL_INIT   | STGFORMER_CHEBYSHEV   | STGFORMER_GEO   |
|:-------|:---------|:------------------------|:------------------------|:--------------------------|:----------------------|:----------------|
| 15 min | MAE      | 1.138                   | **1.126**               | 1.128                     | 1.126                 | 1.128           |
|        | RMSE     | 2.349                   | 2.328                   | **2.319**                 | 2.328                 | 2.327           |
|        | MAPE     | 2.372%                  | 2.361%                  | 2.376%                    | **2.347%**            | 2.363%          |
|        |          |                         |                         |                           |                       |                 |
| 30 min | MAE      | 1.356                   | **1.343**               | 1.345                     | 1.344                 | 1.345           |
|        | RMSE     | 3.001                   | 2.980                   | **2.978**                 | 2.978                 | 2.979           |
|        | MAPE     | 2.934%                  | 2.925%                  | 2.965%                    | **2.911%**            | 2.928%          |
|        |          |                         |                         |                           |                       |                 |
| 1 hour | MAE      | 1.599                   | 1.589                   | **1.587**                 | 1.593                 | 1.595           |
|        | RMSE     | 3.677                   | 3.659                   | **3.655**                 | 3.664                 | 3.663           |
|        | MAPE     | **3.594%**              | 3.608%                  | 3.663%                    | 3.613%                | 3.621%          |

## Temporal

### METR-LA

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_TCN`: TCN temporal mode (causal dilated convolutions)
- `STGFORMER_MAMBA`: Mamba SSM temporal mode (d_state=16, requires CUDA)
- `STGFORMER_MAMBA_FAST`: Optimized Mamba (d_state=8, expand=1)
- `STGFORMER_MLP`: MLP temporal mode

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_TCN   | STGFORMER_MAMBA   | STGFORMER_MAMBA_FAST   | STGFORMER_MLP   |
|:-------|:---------|:------------------------|:----------------|:------------------|:-----------------------|:----------------|
| 15 min | MAE      | 2.616                   | **2.592**       | 2.605             | 2.602                  | 2.626           |
|        | RMSE     | 4.944                   | **4.903**       | 4.949             | 4.957                  | 4.942           |
|        | MAPE     | 6.636%                  | 6.631%          | 6.798%            | **6.606%**             | 6.694%          |
|        |          |                         |                 |                   |                        |                 |
| 30 min | MAE      | 2.862                   | **2.832**       | 2.847             | 2.845                  | 2.876           |
|        | RMSE     | 5.630                   | **5.590**       | 5.631             | 5.662                  | 5.642           |
|        | MAPE     | 7.533%                  | 7.530%          | 7.712%            | **7.523%**             | 7.592%          |
|        |          |                         |                 |                   |                        |                 |
| 1 hour | MAE      | 3.167                   | **3.135**       | 3.143             | 3.151                  | 3.187           |
|        | RMSE     | 6.440                   | **6.387**       | 6.430             | 6.491                  | 6.469           |
|        | MAPE     | 8.683%                  | **8.632%**      | 8.749%            | 8.668%                 | 8.719%          |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_TCN`: TCN temporal mode (causal dilated convolutions)
- `STGFORMER_DEPTHWISE`: Depthwise separable conv temporal mode
- `STGFORMER_MAMBA`: Mamba SSM temporal mode (d_state=16, requires CUDA)
- `STGFORMER_MAMBA_FAST`: Optimized Mamba (d_state=8, expand=1)

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_TCN   | STGFORMER_DEPTHWISE   | STGFORMER_MAMBA   | STGFORMER_MAMBA_FAST   |
|:-------|:---------|:------------------------|:----------------|:----------------------|:------------------|:-----------------------|
| 15 min | MAE      | 1.138                   | **1.134**       | 1.140                 | 1.141             | 1.146                  |
|        | RMSE     | 2.349                   | **2.298**       | 2.322                 | 2.338             | 2.362                  |
|        | MAPE     | 2.372%                  | **2.355%**      | 2.373%                | 2.380%            | 2.370%                 |
|        |          |                         |                 |                       |                   |                        |
| 30 min | MAE      | 1.356                   | 1.358           | **1.354**             | 1.357             | 1.364                  |
|        | RMSE     | 3.001                   | **2.952**       | 2.966                 | 2.991             | 3.005                  |
|        | MAPE     | 2.934%                  | 2.937%          | **2.927%**            | 2.943%            | 2.934%                 |
|        |          |                         |                 |                       |                   |                        |
| 1 hour | MAE      | 1.599                   | 1.603           | 1.602                 | **1.598**         | 1.603                  |
|        | RMSE     | 3.677                   | **3.636**       | 3.659                 | 3.660             | 3.657                  |
|        | MAPE     | **3.594%**              | 3.617%          | 3.614%                | 3.601%            | 3.617%                 |

## Pretraining

### METR-LA

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_PRETRAIN_IMPUTE`: Masked node pretraining (5+5 curriculum) with imputation before fine-tuning
- `STGFORMER_PRETRAIN_STAGE2ONLY_NORM`: Stage 2 only pretraining on NORMALIZED data (10 epochs per-node masking)
- `STGFORMER_PRETRAIN_STAGE1ONLY`: Stage 1 only pretraining (10 epochs per-timestep masking)
- `STGFORMER_PRETRAIN_STAGE1ONLY_NORM`: Stage 1 only pretraining on NORMALIZED data (10 epochs per-timestep masking)

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_PRETRAIN_IMPUTE   | STGFORMER_PRETRAIN_STAGE2ONLY_NORM   | STGFORMER_PRETRAIN_STAGE1ONLY   | STGFORMER_PRETRAIN_STAGE1ONLY_NORM   |
|:-------|:---------|:------------------------|:----------------------------|:-------------------------------------|:--------------------------------|:-------------------------------------|
| 15 min | MAE      | **2.616**               | 2.641                       | 2.624                                | 2.654                           | 2.642                                |
|        | RMSE     | 4.944                   | 5.102                       | **4.912**                            | 5.035                           | 4.997                                |
|        | MAPE     | 6.636%                  | **6.549%**                  | 6.605%                               | 6.747%                          | 6.667%                               |
|        |          |                         |                             |                                      |                                 |                                      |
| 30 min | MAE      | **2.862**               | 2.871                       | 2.872                                | 2.899                           | 2.901                                |
|        | RMSE     | 5.630                   | 5.737                       | **5.587**                            | 5.725                           | 5.722                                |
|        | MAPE     | 7.533%                  | **7.408%**                  | 7.449%                               | 7.606%                          | 7.621%                               |
|        |          |                         |                             |                                      |                                 |                                      |
| 1 hour | MAE      | 3.167                   | **3.162**                   | 3.203                                | 3.215                           | 3.231                                |
|        | RMSE     | 6.440                   | 6.473                       | **6.411**                            | 6.547                           | 6.577                                |
|        | MAPE     | 8.683%                  | 8.599%                      | **8.498%**                           | 8.763%                          | 8.833%                               |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_PRETRAIN_STAGE2ONLY_IMPUTE`: Stage 2 only pretraining with imputation
- `STGFORMER_PRETRAIN_STAGE2ONLY_NORM_IMPUTE`: Stage 2 only pretraining on NORMALIZED data with imputation (loads pretrained checkpoint)
- `STGFORMER_PRETRAIN_STAGE1ONLY`: Stage 1 only pretraining (10 epochs per-timestep masking)
- `STGFORMER_PRETRAIN_STAGE2ONLY`: Stage 2 only pretraining (10 epochs per-node masking)

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_PRETRAIN_STAGE2ONLY_IMPUTE   | STGFORMER_PRETRAIN_STAGE2ONLY_NORM_IMPUTE   | STGFORMER_PRETRAIN_STAGE1ONLY   | STGFORMER_PRETRAIN_STAGE2ONLY   |
|:-------|:---------|:------------------------|:---------------------------------------|:--------------------------------------------|:--------------------------------|:--------------------------------|
| 15 min | MAE      | **1.138**               | 1.144                                  | 1.141                                       | 1.157                           | 1.152                           |
|        | RMSE     | **2.349**               | 2.378                                  | 2.363                                       | 2.388                           | 2.364                           |
|        | MAPE     | 2.372%                  | 2.375%                                 | **2.366%**                                  | 2.408%                          | 2.428%                          |
|        |          |                         |                                        |                                             |                                 |                                 |
| 30 min | MAE      | **1.356**               | 1.357                                  | 1.358                                       | 1.377                           | 1.379                           |
|        | RMSE     | 3.001                   | 3.021                                  | **2.999**                                   | 3.033                           | 3.040                           |
|        | MAPE     | 2.934%                  | **2.933%**                             | 2.934%                                      | 2.984%                          | 3.035%                          |
|        |          |                         |                                        |                                             |                                 |                                 |
| 1 hour | MAE      | 1.599                   | **1.597**                              | 1.604                                       | 1.620                           | 1.635                           |
|        | RMSE     | 3.677                   | 3.689                                  | **3.659**                                   | 3.712                           | 3.746                           |
|        | MAPE     | **3.594%**              | 3.599%                                 | 3.601%                                      | 3.665%                          | 3.742%                          |

## Cheb Tcn Extensions

### METR-LA

**Experiment descriptions**
- `STGFORMER_CHEB_TCN`: Chebyshev propagation + TCN temporal mode
- `STGFORMER_CHEB_TCN_DOW`: Chebyshev+TCN with DOW embeddings
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING`: Chebyshev+TCN+Xavier+DOW excluding missing values from normalization
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=8

| T      | Metric   | STGFORMER_CHEB_TCN   | STGFORMER_CHEB_TCN_DOW   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8   |
|:-------|:---------|:---------------------|:-------------------------|:----------------------------------------------------|:------------------------------------------------|:---------------------------------------------------|
| 15 min | MAE      | 2.570                | **2.563**                | 2.569                                               | 2.569                                           | 2.571                                              |
|        | RMSE     | 4.818                | 4.802                    | **4.789**                                           | 4.791                                           | 4.798                                              |
|        | MAPE     | 6.486%               | **6.470%**               | 6.472%                                              | 6.476%                                          | 6.485%                                             |
|        |          |                      |                          |                                                     |                                                 |                                                    |
| 30 min | MAE      | 2.809                | **2.795**                | 2.797                                               | 2.798                                           | 2.800                                              |
|        | RMSE     | 5.500                | 5.485                    | **5.473**                                           | 5.477                                           | 5.488                                              |
|        | MAPE     | 7.368%               | 7.347%                   | **7.339%**                                          | 7.347%                                          | 7.361%                                             |
|        |          |                      |                          |                                                     |                                                 |                                                    |
| 1 hour | MAE      | 3.117                | 3.081                    | **3.077**                                           | 3.078                                           | 3.082                                              |
|        | RMSE     | 6.325                | 6.279                    | **6.271**                                           | 6.278                                           | 6.293                                              |
|        | MAPE     | 8.542%               | 8.462%                   | **8.446%**                                          | 8.461%                                          | 8.480%                                             |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_CHEB_TCN`: Chebyshev propagation + TCN temporal mode
- `STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE2_NORM_IMPUTE`: Chebyshev+TCN+Xavier+DOW with stage2-only normalized pretraining + imputation
- `STGFORMER_CHEB_TCN_XAVIER_DOW`: Chebyshev+TCN with Xavier initialization and DOW embeddings
- `STGFORMER_CHEB_TCN_EXCLUDE_MISSING`: Chebyshev+TCN excluding missing values from normalization
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]

| T      | Metric   | STGFORMER_CHEB_TCN   | STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE2_NORM_IMPUTE   | STGFORMER_CHEB_TCN_XAVIER_DOW   | STGFORMER_CHEB_TCN_EXCLUDE_MISSING   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   |
|:-------|:---------|:---------------------|:---------------------------------------------------|:--------------------------------|:-------------------------------------|:----------------------------------------------------|
| 15 min | MAE      | 1.125                | **1.113**                                          | 1.123                           | 1.121                                | 1.125                                               |
|        | RMSE     | 2.293                | **2.235**                                          | 2.280                           | 2.258                                | 2.257                                               |
|        | MAPE     | 2.328%               | **2.265%**                                         | 2.322%                          | 2.269%                               | 2.291%                                              |
|        |          |                      |                                                    |                                 |                                      |                                                     |
| 30 min | MAE      | 1.343                | **1.328**                                          | 1.334                           | 1.337                                | 1.338                                               |
|        | RMSE     | 2.934                | **2.891**                                          | 2.912                           | 2.903                                | 2.900                                               |
|        | MAPE     | 2.893%               | **2.839%**                                         | 2.866%                          | 2.840%                               | 2.844%                                              |
|        |          |                      |                                                    |                                 |                                      |                                                     |
| 1 hour | MAE      | 1.591                | 1.577                                              | **1.570**                       | 1.581                                | 1.577                                               |
|        | RMSE     | 3.624                | 3.598                                              | **3.575**                       | 3.592                                | 3.578                                               |
|        | MAPE     | 3.581%               | 3.530%                                             | **3.521%**                      | 3.551%                               | 3.524%                                              |

## Ablation

### METR-LA

**Experiment descriptions**
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]
- `ABL_NO_XAVIER`: Ablation: Remove Xavier initialization from final model
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING`: Chebyshev+TCN+Xavier+DOW excluding missing values from normalization
- `ABL_NO_DOW`: Ablation: Remove DOW embeddings from final model
- `ABL_NO_CHEB`: Ablation: Replace Chebyshev with standard graph convolution from final model

| T      | Metric   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   | ABL_NO_XAVIER   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING   | ABL_NO_DOW   | ABL_NO_CHEB   |
|:-------|:---------|:----------------------------------------------------|:----------------|:------------------------------------------------|:-------------|:--------------|
| 15 min | MAE      | 2.569                                               | **2.561**       | 2.569                                           | 2.572        | 2.598         |
|        | RMSE     | 4.789                                               | 4.787           | 4.791                                           | **4.764**    | 4.906         |
|        | MAPE     | 6.472%                                              | 6.470%          | 6.476%                                          | **6.435%**   | 6.600%        |
|        |          |                                                     |                 |                                                 |              |               |
| 30 min | MAE      | 2.797                                               | **2.788**       | 2.798                                           | 2.813        | 2.830         |
|        | RMSE     | 5.473                                               | 5.466           | 5.477                                           | **5.458**    | 5.597         |
|        | MAPE     | 7.339%                                              | 7.346%          | 7.347%                                          | **7.298%**   | 7.504%        |
|        |          |                                                     |                 |                                                 |              |               |
| 1 hour | MAE      | 3.077                                               | **3.070**       | 3.078                                           | 3.126        | 3.110         |
|        | RMSE     | 6.271                                               | **6.264**       | 6.278                                           | 6.303        | 6.383         |
|        | MAPE     | 8.446%                                              | 8.487%          | 8.461%                                          | **8.433%**   | 8.628%        |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]
- `ABL_NO_XAVIER`: Ablation: Remove Xavier initialization from final model
- `ABL_NO_EXCLUDE_MISSING`: Ablation: Remove ExcludeMissing normalization from final model
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING`: Chebyshev+TCN+Xavier+DOW excluding missing values from normalization
- `ABL_NO_DOW`: Ablation: Remove DOW embeddings from final model

| T      | Metric   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   | ABL_NO_XAVIER   | ABL_NO_EXCLUDE_MISSING   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING   | ABL_NO_DOW   |
|:-------|:---------|:----------------------------------------------------|:----------------|:-------------------------|:------------------------------------------------|:-------------|
| 15 min | MAE      | 1.125                                               | **1.123**       | 1.125                    | 1.125                                           | 1.124        |
|        | RMSE     | **2.257**                                           | 2.260           | 2.274                    | 2.259                                           | 2.261        |
|        | MAPE     | 2.291%                                              | **2.283%**      | 2.331%                   | 2.290%                                          | 2.288%       |
|        |          |                                                     |                 |                          |                                                 |              |
| 30 min | MAE      | 1.338                                               | **1.336**       | 1.338                    | 1.338                                           | 1.339        |
|        | RMSE     | 2.900                                               | **2.898**       | 2.914                    | 2.902                                           | 2.914        |
|        | MAPE     | 2.844%                                              | **2.836%**      | 2.882%                   | 2.843%                                          | 2.853%       |
|        |          |                                                     |                 |                          |                                                 |              |
| 1 hour | MAE      | 1.577                                               | **1.576**       | 1.577                    | 1.577                                           | 1.584        |
|        | RMSE     | 3.578                                               | **3.569**       | 3.593                    | 3.577                                           | 3.609        |
|        | MAPE     | 3.524%                                              | 3.524%          | 3.545%                   | **3.521%**                                      | 3.562%       |
