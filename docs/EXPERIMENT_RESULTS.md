# Experiment Results

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
- `STGFORMER_GEO`: Geographic (pre-computed) graph
- `STGFORMER_HYBRID`: Hybrid graph (geographic + learned)
- `STGFORMER_SPECTRAL_INIT`: Learned graph initialized from Laplacian eigenvectors
- `STGFORMER_CHEBYSHEV`: Chebyshev polynomial propagation
- `STGFORMER_GEO_CHEB`: Geographic graph + Chebyshev polynomial propagation
- `STGFORMER_HYBRID_CHEB`: Hybrid graph + Chebyshev polynomial propagation

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_GEO   | STGFORMER_HYBRID   | STGFORMER_SPECTRAL_INIT   | STGFORMER_CHEBYSHEV   | STGFORMER_GEO_CHEB   | STGFORMER_HYBRID_CHEB   |
|:-------|:---------|:------------------------|:----------------|:-------------------|:--------------------------|:----------------------|:---------------------|:------------------------|
| 15 min | MAE      | 2.616                   | 2.601           | 2.595              | 2.601                     | 2.594                 | **2.592**            | 2.597                   |
|        | RMSE     | 4.944                   | 4.959           | 4.936              | 4.929                     | **4.898**             | 4.921                | 4.898                   |
|        | MAPE     | 6.636%                  | 6.780%          | 6.729%             | **6.599%**                | 6.684%                | 6.681%               | 6.677%                  |
|        |          |                         |                 |                    |                           |                       |                      |                         |
| 30 min | MAE      | 2.862                   | 2.847           | 2.837              | 2.846                     | 2.836                 | **2.835**            | 2.839                   |
|        | RMSE     | 5.630                   | 5.677           | 5.644              | 5.628                     | 5.609                 | 5.633                | **5.604**               |
|        | MAPE     | 7.533%                  | 7.748%          | 7.686%             | **7.486%**                | 7.575%                | 7.584%               | 7.540%                  |
|        |          |                         |                 |                    |                           |                       |                      |                         |
| 1 hour | MAE      | 3.167                   | 3.149           | **3.136**          | 3.152                     | 3.140                 | 3.136                | 3.143                   |
|        | RMSE     | 6.440                   | 6.504           | 6.455              | 6.438                     | 6.430                 | 6.432                | **6.419**               |
|        | MAPE     | 8.683%                  | 8.948%          | 8.868%             | **8.608%**                | 8.713%                | 8.734%               | 8.656%                  |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_GEO`: Geographic (pre-computed) graph
- `STGFORMER_HYBRID`: Hybrid graph (geographic + learned)
- `STGFORMER_SPECTRAL_INIT`: Learned graph initialized from Laplacian eigenvectors
- `STGFORMER_CHEBYSHEV`: Chebyshev polynomial propagation
- `STGFORMER_GEO_CHEB`: Geographic graph + Chebyshev polynomial propagation
- `STGFORMER_HYBRID_CHEB`: Hybrid graph + Chebyshev polynomial propagation

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_GEO   | STGFORMER_HYBRID   | STGFORMER_SPECTRAL_INIT   | STGFORMER_CHEBYSHEV   | STGFORMER_GEO_CHEB   | STGFORMER_HYBRID_CHEB   |
|:-------|:---------|:------------------------|:----------------|:-------------------|:--------------------------|:----------------------|:---------------------|:------------------------|
| 15 min | MAE      | 1.138                   | 1.128           | 1.138              | 1.128                     | 1.126                 | 1.138                | **1.126**               |
|        | RMSE     | 2.349                   | 2.327           | 2.356              | **2.319**                 | 2.328                 | 2.342                | 2.328                   |
|        | MAPE     | 2.372%                  | 2.363%          | 2.398%             | 2.376%                    | **2.347%**            | 2.399%               | 2.361%                  |
|        |          |                         |                 |                    |                           |                       |                      |                         |
| 30 min | MAE      | 1.356                   | 1.345           | 1.349              | 1.345                     | 1.344                 | 1.355                | **1.343**               |
|        | RMSE     | 3.001                   | 2.979           | 2.989              | **2.978**                 | 2.978                 | 2.996                | 2.980                   |
|        | MAPE     | 2.934%                  | 2.928%          | 2.944%             | 2.965%                    | **2.911%**            | 2.985%               | 2.925%                  |
|        |          |                         |                 |                    |                           |                       |                      |                         |
| 1 hour | MAE      | 1.599                   | 1.595           | 1.594              | **1.587**                 | 1.593                 | 1.599                | 1.589                   |
|        | RMSE     | 3.677                   | 3.663           | 3.658              | **3.655**                 | 3.664                 | 3.675                | 3.659                   |
|        | MAPE     | **3.594%**              | 3.621%          | 3.633%             | 3.663%                    | 3.613%                | 3.674%               | 3.608%                  |

## Temporal

### METR-LA

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_MLP`: MLP temporal mode
- `STGFORMER_TCN`: TCN temporal mode (causal dilated convolutions)
- `STGFORMER_DEPTHWISE`: Depthwise separable conv temporal mode
- `STGFORMER_MAMBA_FAST`: Optimized Mamba (d_state=8, expand=1)
- `STGFORMER_MAMBA`: Mamba SSM temporal mode (d_state=16, requires CUDA)

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_MLP   | STGFORMER_TCN   | STGFORMER_DEPTHWISE   | STGFORMER_MAMBA_FAST   | STGFORMER_MAMBA   |
|:-------|:---------|:------------------------|:----------------|:----------------|:----------------------|:-----------------------|:------------------|
| 15 min | MAE      | 2.616                   | 2.626           | **2.592**       | 2.636                 | 2.602                  | 2.605             |
|        | RMSE     | 4.944                   | 4.942           | **4.903**       | 4.910                 | 4.957                  | 4.949             |
|        | MAPE     | 6.636%                  | 6.694%          | 6.631%          | 6.813%                | **6.606%**             | 6.798%            |
|        |          |                         |                 |                 |                       |                        |                   |
| 30 min | MAE      | 2.862                   | 2.876           | **2.832**       | 2.885                 | 2.845                  | 2.847             |
|        | RMSE     | 5.630                   | 5.642           | **5.590**       | 5.603                 | 5.662                  | 5.631             |
|        | MAPE     | 7.533%                  | 7.592%          | 7.530%          | 7.777%                | **7.523%**             | 7.712%            |
|        |          |                         |                 |                 |                       |                        |                   |
| 1 hour | MAE      | 3.167                   | 3.187           | **3.135**       | 3.195                 | 3.151                  | 3.143             |
|        | RMSE     | 6.440                   | 6.469           | **6.387**       | 6.440                 | 6.491                  | 6.430             |
|        | MAPE     | 8.683%                  | 8.719%          | **8.632%**      | 8.965%                | 8.668%                 | 8.749%            |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_MLP`: MLP temporal mode
- `STGFORMER_TCN`: TCN temporal mode (causal dilated convolutions)
- `STGFORMER_DEPTHWISE`: Depthwise separable conv temporal mode
- `STGFORMER_MAMBA_FAST`: Optimized Mamba (d_state=8, expand=1)
- `STGFORMER_MAMBA`: Mamba SSM temporal mode (d_state=16, requires CUDA)

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_MLP   | STGFORMER_TCN   | STGFORMER_DEPTHWISE   | STGFORMER_MAMBA_FAST   | STGFORMER_MAMBA   |
|:-------|:---------|:------------------------|:----------------|:----------------|:----------------------|:-----------------------|:------------------|
| 15 min | MAE      | 1.138                   | 1.154           | **1.134**       | 1.140                 | 1.146                  | 1.141             |
|        | RMSE     | 2.349                   | 2.346           | **2.298**       | 2.322                 | 2.362                  | 2.338             |
|        | MAPE     | 2.372%                  | 2.411%          | **2.355%**      | 2.373%                | 2.370%                 | 2.380%            |
|        |          |                         |                 |                 |                       |                        |                   |
| 30 min | MAE      | 1.356                   | 1.378           | 1.358           | **1.354**             | 1.364                  | 1.357             |
|        | RMSE     | 3.001                   | 3.025           | **2.952**       | 2.966                 | 3.005                  | 2.991             |
|        | MAPE     | 2.934%                  | 2.996%          | 2.937%          | **2.927%**            | 2.934%                 | 2.943%            |
|        |          |                         |                 |                 |                       |                        |                   |
| 1 hour | MAE      | 1.599                   | 1.630           | 1.603           | 1.602                 | 1.603                  | **1.598**         |
|        | RMSE     | 3.677                   | 3.740           | **3.636**       | 3.659                 | 3.657                  | 3.660             |
|        | MAPE     | **3.594%**              | 3.681%          | 3.617%          | 3.614%                | 3.617%                 | 3.601%            |

## Pretraining

### METR-LA

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_PRETRAIN`: Masked node pretraining (5+5 curriculum) + forecasting fine-tune
- `STGFORMER_PRETRAIN_IMPUTE`: Masked node pretraining (5+5 curriculum) with imputation before fine-tuning
- `STGFORMER_PRETRAIN_IMPUTE_NORM`: Masked node pretraining (5+5 curriculum) with imputation on NORMALIZED data
- `STGFORMER_PRETRAIN_IMPUTE_NORM_LR0.0003`: Pretrain+impute normalized with lower fine-tuning LR (0.0003)
- `STGFORMER_PRETRAIN_STAGE1ONLY`: Stage 1 only pretraining (10 epochs per-timestep masking)
- `STGFORMER_PRETRAIN_STAGE1ONLY_IMPUTE`: Stage 1 only pretraining with imputation
- `STGFORMER_PRETRAIN_STAGE1ONLY_NORM`: Stage 1 only pretraining on NORMALIZED data (10 epochs per-timestep masking)
- `STGFORMER_PRETRAIN_STAGE1ONLY_NORM_IMPUTE`: Stage 1 only pretraining on NORMALIZED data with imputation (loads pretrained checkpoint)
- `STGFORMER_PRETRAIN_STAGE2ONLY`: Stage 2 only pretraining (10 epochs per-node masking)
- `STGFORMER_PRETRAIN_STAGE2ONLY_IMPUTE`: Stage 2 only pretraining with imputation
- `STGFORMER_PRETRAIN_STAGE2ONLY_NORM`: Stage 2 only pretraining on NORMALIZED data (10 epochs per-node masking)
- `STGFORMER_PRETRAIN_STAGE2ONLY_NORM_IMPUTE`: Stage 2 only pretraining on NORMALIZED data with imputation (loads pretrained checkpoint)

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_PRETRAIN   | STGFORMER_PRETRAIN_IMPUTE   | STGFORMER_PRETRAIN_IMPUTE_NORM   | STGFORMER_PRETRAIN_IMPUTE_NORM_LR0.0003   | STGFORMER_PRETRAIN_STAGE1ONLY   | STGFORMER_PRETRAIN_STAGE1ONLY_IMPUTE   | STGFORMER_PRETRAIN_STAGE1ONLY_NORM   | STGFORMER_PRETRAIN_STAGE1ONLY_NORM_IMPUTE   | STGFORMER_PRETRAIN_STAGE2ONLY   | STGFORMER_PRETRAIN_STAGE2ONLY_IMPUTE   | STGFORMER_PRETRAIN_STAGE2ONLY_NORM   | STGFORMER_PRETRAIN_STAGE2ONLY_NORM_IMPUTE   |
|:-------|:---------|:------------------------|:---------------------|:----------------------------|:---------------------------------|:------------------------------------------|:--------------------------------|:---------------------------------------|:-------------------------------------|:--------------------------------------------|:--------------------------------|:---------------------------------------|:-------------------------------------|:--------------------------------------------|
| 15 min | MAE      | **2.616**               | 2.670                | 2.641                       | 2.722                            | 2.908                                     | 2.654                           | 2.813                                  | 2.642                                | 2.795                                       | 2.648                           | 2.739                                  | 2.624                                | 3.024                                       |
|        | RMSE     | 4.944                   | 5.029                | 5.102                       | 5.243                            | 5.874                                     | 5.035                           | 5.773                                  | 4.997                                | 5.532                                       | 4.982                           | 5.254                                  | **4.912**                            | 6.602                                       |
|        | MAPE     | 6.636%                  | 6.717%               | **6.549%**                  | 6.587%                           | 6.832%                                    | 6.747%                          | 6.924%                                 | 6.667%                               | 6.894%                                      | 6.719%                          | 6.851%                                 | 6.605%                               | 7.151%                                      |
|        |          |                         |                      |                             |                                  |                                           |                                 |                                        |                                      |                                             |                                 |                                        |                                      |                                             |
| 30 min | MAE      | **2.862**               | 2.905                | 2.871                       | 2.959                            | 3.281                                     | 2.899                           | 3.137                                  | 2.901                                | 3.102                                       | 2.904                           | 3.008                                  | 2.872                                | 3.431                                       |
|        | RMSE     | 5.630                   | 5.683                | 5.737                       | 5.849                            | 6.841                                     | 5.725                           | 6.609                                  | 5.722                                | 6.298                                       | 5.668                           | 5.939                                  | **5.587**                            | 7.664                                       |
|        | MAPE     | 7.533%                  | 7.586%               | 7.408%                      | **7.358%**                       | 7.811%                                    | 7.606%                          | 7.908%                                 | 7.621%                               | 7.910%                                      | 7.643%                          | 7.770%                                 | 7.449%                               | 8.255%                                      |
|        |          |                         |                      |                             |                                  |                                           |                                 |                                        |                                      |                                             |                                 |                                        |                                      |                                             |
| 1 hour | MAE      | 3.167                   | 3.214                | **3.162**                   | 3.271                            | 3.773                                     | 3.215                           | 3.514                                  | 3.231                                | 3.483                                       | 3.228                           | 3.368                                  | 3.203                                | 3.945                                       |
|        | RMSE     | 6.440                   | 6.485                | 6.473                       | 6.600                            | 7.977                                     | 6.547                           | 7.433                                  | 6.577                                | 7.163                                       | 6.498                           | 6.782                                  | **6.411**                            | 8.762                                       |
|        | MAPE     | 8.683%                  | 8.747%               | 8.599%                      | **8.413%**                       | 9.125%                                    | 8.763%                          | 9.143%                                 | 8.833%                               | 9.189%                                      | 8.859%                          | 9.016%                                 | 8.498%                               | 9.647%                                      |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_BS200_SHORT`: Baseline with batch_size=200, 20 epochs, early_stop=5
- `STGFORMER_PRETRAIN`: Masked node pretraining (5+5 curriculum) + forecasting fine-tune
- `STGFORMER_PRETRAIN_IMPUTE`: Masked node pretraining (5+5 curriculum) with imputation before fine-tuning
- `STGFORMER_PRETRAIN_IMPUTE_NORM`: Masked node pretraining (5+5 curriculum) with imputation on NORMALIZED data
- `STGFORMER_PRETRAIN_IMPUTE_NORM_LR0.0003`: Pretrain+impute normalized with lower fine-tuning LR (0.0003)
- `STGFORMER_PRETRAIN_STAGE1ONLY`: Stage 1 only pretraining (10 epochs per-timestep masking)
- `STGFORMER_PRETRAIN_STAGE1ONLY_IMPUTE`: Stage 1 only pretraining with imputation
- `STGFORMER_PRETRAIN_STAGE1ONLY_NORM`: Stage 1 only pretraining on NORMALIZED data (10 epochs per-timestep masking)
- `STGFORMER_PRETRAIN_STAGE1ONLY_NORM_IMPUTE`: Stage 1 only pretraining on NORMALIZED data with imputation (loads pretrained checkpoint)
- `STGFORMER_PRETRAIN_STAGE2ONLY`: Stage 2 only pretraining (10 epochs per-node masking)
- `STGFORMER_PRETRAIN_STAGE2ONLY_IMPUTE`: Stage 2 only pretraining with imputation
- `STGFORMER_PRETRAIN_STAGE2ONLY_NORM`: Stage 2 only pretraining on NORMALIZED data (10 epochs per-node masking)
- `STGFORMER_PRETRAIN_STAGE2ONLY_NORM_IMPUTE`: Stage 2 only pretraining on NORMALIZED data with imputation (loads pretrained checkpoint)

| T      | Metric   | STGFORMER_BS200_SHORT   | STGFORMER_PRETRAIN   | STGFORMER_PRETRAIN_IMPUTE   | STGFORMER_PRETRAIN_IMPUTE_NORM   | STGFORMER_PRETRAIN_IMPUTE_NORM_LR0.0003   | STGFORMER_PRETRAIN_STAGE1ONLY   | STGFORMER_PRETRAIN_STAGE1ONLY_IMPUTE   | STGFORMER_PRETRAIN_STAGE1ONLY_NORM   | STGFORMER_PRETRAIN_STAGE1ONLY_NORM_IMPUTE   | STGFORMER_PRETRAIN_STAGE2ONLY   | STGFORMER_PRETRAIN_STAGE2ONLY_IMPUTE   | STGFORMER_PRETRAIN_STAGE2ONLY_NORM   | STGFORMER_PRETRAIN_STAGE2ONLY_NORM_IMPUTE   |
|:-------|:---------|:------------------------|:---------------------|:----------------------------|:---------------------------------|:------------------------------------------|:--------------------------------|:---------------------------------------|:-------------------------------------|:--------------------------------------------|:--------------------------------|:---------------------------------------|:-------------------------------------|:--------------------------------------------|
| 15 min | MAE      | **1.138**               | 1.293                | 1.191                       | 1.287                            | 1.153                                     | 1.157                           | 1.167                                  | 1.155                                | 1.156                                       | 1.152                           | 1.144                                  | 1.171                                | 1.141                                       |
|        | RMSE     | 2.349                   | 2.531                | 2.403                       | 2.575                            | 2.401                                     | 2.388                           | 2.367                                  | **2.335**                            | 2.369                                       | 2.364                           | 2.378                                  | 2.355                                | 2.363                                       |
|        | MAPE     | 2.372%                  | 2.673%               | 2.459%                      | 2.744%                           | 2.389%                                    | 2.408%                          | 2.408%                                 | 2.391%                               | 2.413%                                      | 2.428%                          | 2.375%                                 | 2.423%                               | **2.366%**                                  |
|        |          |                         |                      |                             |                                  |                                           |                                 |                                        |                                      |                                             |                                 |                                        |                                      |                                             |
| 30 min | MAE      | **1.356**               | 1.508                | 1.397                       | 1.478                            | 1.380                                     | 1.377                           | 1.388                                  | 1.384                                | 1.381                                       | 1.379                           | 1.357                                  | 1.388                                | 1.358                                       |
|        | RMSE     | 3.001                   | 3.157                | 3.013                       | 3.158                            | 3.053                                     | 3.033                           | **2.999**                              | 3.017                                | 3.041                                       | 3.040                           | 3.021                                  | 3.008                                | 2.999                                       |
|        | MAPE     | 2.934%                  | 3.226%               | 3.001%                      | 3.233%                           | 2.976%                                    | 2.984%                          | 2.996%                                 | 2.978%                               | 2.990%                                      | 3.035%                          | **2.933%**                             | 2.984%                               | 2.934%                                      |
|        |          |                         |                      |                             |                                  |                                           |                                 |                                        |                                      |                                             |                                 |                                        |                                      |                                             |
| 1 hour | MAE      | 1.599                   | 1.772                | 1.630                       | 1.706                            | 1.639                                     | 1.620                           | 1.631                                  | 1.647                                | 1.636                                       | 1.635                           | **1.597**                              | 1.632                                | 1.604                                       |
|        | RMSE     | 3.677                   | 3.853                | **3.654**                   | 3.775                            | 3.742                                     | 3.712                           | 3.667                                  | 3.734                                | 3.742                                       | 3.746                           | 3.689                                  | 3.678                                | 3.659                                       |
|        | MAPE     | **3.594%**              | 3.943%               | 3.653%                      | 3.879%                           | 3.673%                                    | 3.665%                          | 3.706%                                 | 3.679%                               | 3.688%                                      | 3.742%                          | 3.599%                                 | 3.638%                               | 3.601%                                      |

## Cheb Tcn Extensions

### METR-LA

**Experiment descriptions**
- `STGFORMER_CHEB_TCN`: Chebyshev propagation + TCN temporal mode
- `STGFORMER_CHEB_TCN_EXCLUDE_MISSING`: Chebyshev+TCN excluding missing values from normalization
- `STGFORMER_CHEB_TCN_XAVIER`: Chebyshev+TCN with Xavier initialization
- `STGFORMER_CHEB_TCN_DOW`: Chebyshev+TCN with DOW embeddings
- `STGFORMER_CHEB_TCN_XAVIER_DOW`: Chebyshev+TCN with Xavier initialization and DOW embeddings
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING`: Chebyshev+TCN+Xavier+DOW excluding missing values from normalization
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=8
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]
- `STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE1_NORM_IMPUTE`: Chebyshev+TCN+Xavier+DOW with stage1-only normalized pretraining + imputation
- `STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE2_NORM_IMPUTE`: Chebyshev+TCN+Xavier+DOW with stage2-only normalized pretraining + imputation
- `STGFORMER_CHEB_TCN_XAVIER_DOW_PRETRAIN_NORM_IMPUTE`: Chebyshev+TCN+Xavier+DOW with full normalized pretraining + imputation

| T      | Metric   | STGFORMER_CHEB_TCN   | STGFORMER_CHEB_TCN_EXCLUDE_MISSING   | STGFORMER_CHEB_TCN_XAVIER   | STGFORMER_CHEB_TCN_DOW   | STGFORMER_CHEB_TCN_XAVIER_DOW   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   | STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE1_NORM_IMPUTE   | STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE2_NORM_IMPUTE   | STGFORMER_CHEB_TCN_XAVIER_DOW_PRETRAIN_NORM_IMPUTE   |
|:-------|:---------|:---------------------|:-------------------------------------|:----------------------------|:-------------------------|:--------------------------------|:------------------------------------------------|:---------------------------------------------------|:----------------------------------------------------|:---------------------------------------------------|:---------------------------------------------------|:-----------------------------------------------------|
| 15 min | MAE      | 2.570                | 2.568                                | 2.585                       | **2.563**                | 2.756                           | 2.569                                           | 2.571                                              | 2.569                                               | 2.653                                              | 2.811                                              | 2.775                                                |
|        | RMSE     | 4.818                | 4.812                                | 4.837                       | 4.802                    | 5.185                           | 4.791                                           | 4.798                                              | **4.789**                                           | 5.062                                              | 5.820                                              | 5.261                                                |
|        | MAPE     | 6.486%               | 6.495%                               | 6.537%                      | **6.470%**               | 7.180%                          | 6.476%                                          | 6.485%                                             | 6.472%                                              | 6.740%                                             | 6.740%                                             | 6.858%                                               |
|        |          |                      |                                      |                             |                          |                                 |                                                 |                                                    |                                                     |                                                    |                                                    |                                                      |
| 30 min | MAE      | 2.809                | 2.810                                | 2.828                       | **2.795**                | 3.027                           | 2.798                                           | 2.800                                              | 2.797                                               | 2.893                                              | 3.142                                              | 3.009                                                |
|        | RMSE     | 5.500                | 5.511                                | 5.528                       | 5.485                    | 5.939                           | 5.477                                           | 5.488                                              | **5.473**                                           | 5.737                                              | 6.715                                              | 5.886                                                |
|        | MAPE     | 7.368%               | 7.401%                               | 7.440%                      | 7.347%                   | 8.198%                          | 7.347%                                          | 7.361%                                             | **7.339%**                                          | 7.666%                                             | 7.728%                                             | 7.694%                                               |
|        |          |                      |                                      |                             |                          |                                 |                                                 |                                                    |                                                     |                                                    |                                                    |                                                      |
| 1 hour | MAE      | 3.117                | 3.124                                | 3.143                       | 3.081                    | 3.431                           | 3.078                                           | 3.082                                              | **3.077**                                           | 3.191                                              | 3.516                                              | 3.301                                                |
|        | RMSE     | 6.325                | 6.360                                | 6.374                       | 6.279                    | 6.940                           | 6.278                                           | 6.293                                              | **6.271**                                           | 6.553                                              | 7.621                                              | 6.634                                                |
|        | MAPE     | 8.542%               | 8.593%                               | 8.639%                      | 8.462%                   | 9.859%                          | 8.461%                                          | 8.480%                                             | **8.446%**                                          | 8.924%                                             | 8.928%                                             | 8.817%                                               |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_CHEB_TCN`: Chebyshev propagation + TCN temporal mode
- `STGFORMER_CHEB_TCN_EXCLUDE_MISSING`: Chebyshev+TCN excluding missing values from normalization
- `STGFORMER_CHEB_TCN_XAVIER`: Chebyshev+TCN with Xavier initialization
- `STGFORMER_CHEB_TCN_DOW`: Chebyshev+TCN with DOW embeddings
- `STGFORMER_CHEB_TCN_XAVIER_DOW`: Chebyshev+TCN with Xavier initialization and DOW embeddings
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING`: Chebyshev+TCN+Xavier+DOW excluding missing values from normalization
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=8
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]
- `STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE1_NORM_IMPUTE`: Chebyshev+TCN+Xavier+DOW with stage1-only normalized pretraining + imputation
- `STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE2_NORM_IMPUTE`: Chebyshev+TCN+Xavier+DOW with stage2-only normalized pretraining + imputation
- `STGFORMER_CHEB_TCN_XAVIER_DOW_PRETRAIN_NORM_IMPUTE`: Chebyshev+TCN+Xavier+DOW with full normalized pretraining + imputation

| T      | Metric   | STGFORMER_CHEB_TCN   | STGFORMER_CHEB_TCN_EXCLUDE_MISSING   | STGFORMER_CHEB_TCN_XAVIER   | STGFORMER_CHEB_TCN_DOW   | STGFORMER_CHEB_TCN_XAVIER_DOW   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   | STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE1_NORM_IMPUTE   | STGFORMER_CHEB_TCN_XAVIER_DOW_STAGE2_NORM_IMPUTE   | STGFORMER_CHEB_TCN_XAVIER_DOW_PRETRAIN_NORM_IMPUTE   |
|:-------|:---------|:---------------------|:-------------------------------------|:----------------------------|:-------------------------|:--------------------------------|:------------------------------------------------|:---------------------------------------------------|:----------------------------------------------------|:---------------------------------------------------|:---------------------------------------------------|:-----------------------------------------------------|
| 15 min | MAE      | 1.125                | 1.121                                | 1.124                       | 1.125                    | 1.123                           | 1.125                                           | 1.127                                              | 1.125                                               | 1.125                                              | **1.113**                                          | 1.144                                                |
|        | RMSE     | 2.293                | 2.258                                | 2.275                       | 2.288                    | 2.280                           | 2.259                                           | 2.267                                              | 2.257                                               | 2.256                                              | **2.235**                                          | 2.288                                                |
|        | MAPE     | 2.328%               | 2.269%                               | 2.327%                      | 2.340%                   | 2.322%                          | 2.290%                                          | 2.295%                                             | 2.291%                                              | 2.280%                                             | **2.265%**                                         | 2.342%                                               |
|        |          |                      |                                      |                             |                          |                                 |                                                 |                                                    |                                                     |                                                    |                                                    |                                                      |
| 30 min | MAE      | 1.343                | 1.337                                | 1.338                       | 1.338                    | 1.334                           | 1.338                                           | 1.341                                              | 1.338                                               | 1.353                                              | **1.328**                                          | 1.392                                                |
|        | RMSE     | 2.934                | 2.903                                | 2.919                       | 2.921                    | 2.912                           | 2.902                                           | 2.914                                              | 2.900                                               | 2.929                                              | **2.891**                                          | 2.992                                                |
|        | MAPE     | 2.893%               | 2.840%                               | 2.883%                      | 2.897%                   | 2.866%                          | 2.843%                                          | 2.857%                                             | 2.844%                                              | 2.906%                                             | **2.839%**                                         | 2.992%                                               |
|        |          |                      |                                      |                             |                          |                                 |                                                 |                                                    |                                                     |                                                    |                                                    |                                                      |
| 1 hour | MAE      | 1.591                | 1.581                                | 1.583                       | 1.577                    | **1.570**                       | 1.577                                           | 1.579                                              | 1.577                                               | 1.604                                              | 1.577                                              | 1.662                                                |
|        | RMSE     | 3.624                | 3.592                                | 3.604                       | 3.591                    | **3.575**                       | 3.577                                           | 3.587                                              | 3.578                                               | 3.645                                              | 3.598                                              | 3.742                                                |
|        | MAPE     | 3.581%               | 3.551%                               | 3.561%                      | 3.570%                   | **3.521%**                      | 3.521%                                          | 3.542%                                             | 3.524%                                              | 3.681%                                             | 3.530%                                             | 3.744%                                               |

## Ablation

### METR-LA

**Experiment descriptions**
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING`: Chebyshev+TCN+Xavier+DOW excluding missing values from normalization
- `ABL_NO_XAVIER`: Ablation: Remove Xavier initialization from final model
- `ABL_NO_DOW`: Ablation: Remove DOW embeddings from final model
- `ABL_NO_EXCLUDE_MISSING`: Ablation: Remove ExcludeMissing normalization from final model
- `ABL_NO_TCN`: Ablation: Replace TCN with standard Transformer in the final model
- `ABL_NO_CHEB`: Ablation: Replace Chebyshev with standard graph convolution from final model
- `ABL_NO_GRAPH`: Ablation: Remove graph propagation (GraphMode.NONE) from final model
- `ABL_NO_TEMPORAL`: Ablation: Remove temporal processing (TemporalMode.NONE) from final model

| T      | Metric   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING   | ABL_NO_XAVIER   | ABL_NO_DOW   | ABL_NO_EXCLUDE_MISSING   | ABL_NO_CHEB   | ABL_NO_GRAPH   | ABL_NO_TEMPORAL   |
|:-------|:---------|:----------------------------------------------------|:------------------------------------------------|:----------------|:-------------|:-------------------------|:--------------|:---------------|:------------------|
| 15 min | MAE      | 2.569                                               | 2.569                                           | **2.561**       | 2.572        | 2.695                    | 2.598         | 2.685          | 2.620             |
|        | RMSE     | 4.789                                               | 4.791                                           | 4.787           | **4.764**    | 5.025                    | 4.906         | 5.046          | 4.908             |
|        | MAPE     | 6.472%                                              | 6.476%                                          | 6.470%          | **6.435%**   | 6.887%                   | 6.600%        | 6.838%         | 6.686%            |
|        |          |                                                     |                                                 |                 |              |                          |               |                |                   |
| 30 min | MAE      | 2.797                                               | 2.798                                           | **2.788**       | 2.813        | 2.953                    | 2.830         | 3.007          | 2.856             |
|        | RMSE     | 5.473                                               | 5.477                                           | 5.466           | **5.458**    | 5.754                    | 5.597         | 5.904          | 5.596             |
|        | MAPE     | 7.339%                                              | 7.347%                                          | 7.346%          | **7.298%**   | 7.869%                   | 7.504%        | 8.067%         | 7.592%            |
|        |          |                                                     |                                                 |                 |              |                          |               |                |                   |
| 1 hour | MAE      | 3.077                                               | 3.078                                           | **3.070**       | 3.126        | 3.299                    | 3.110         | 3.489          | 3.149             |
|        | RMSE     | 6.271                                               | 6.278                                           | **6.264**       | 6.303        | 6.661                    | 6.383         | 7.062          | 6.396             |
|        | MAPE     | 8.446%                                              | 8.461%                                          | 8.487%          | **8.433%**   | 9.346%                   | 8.628%        | 9.984%         | 8.790%            |

### PEMS-BAY

**Experiment descriptions**
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16`: Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]
- `STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING`: Chebyshev+TCN+Xavier+DOW excluding missing values from normalization
- `ABL_NO_XAVIER`: Ablation: Remove Xavier initialization from final model
- `ABL_NO_DOW`: Ablation: Remove DOW embeddings from final model
- `ABL_NO_EXCLUDE_MISSING`: Ablation: Remove ExcludeMissing normalization from final model
- `ABL_NO_TCN`: Ablation: Replace TCN with standard Transformer in the final model
- `ABL_NO_CHEB`: Ablation: Replace Chebyshev with standard graph convolution from final model
- `ABL_NO_GRAPH`: Ablation: Remove graph propagation (GraphMode.NONE) from final model
- `ABL_NO_TEMPORAL`: Ablation: Remove temporal processing (TemporalMode.NONE) from final model

| T      | Metric   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16   | STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING   | ABL_NO_XAVIER   | ABL_NO_DOW   | ABL_NO_EXCLUDE_MISSING   | ABL_NO_CHEB   | ABL_NO_GRAPH   | ABL_NO_TEMPORAL   |
|:-------|:---------|:----------------------------------------------------|:------------------------------------------------|:----------------|:-------------|:-------------------------|:--------------|:---------------|:------------------|
| 15 min | MAE      | 1.125                                               | 1.125                                           | **1.123**       | 1.124        | 1.125                    | 1.131         | 1.158          | 1.180             |
|        | RMSE     | **2.257**                                           | 2.259                                           | 2.260           | 2.261        | 2.274                    | 2.288         | 2.377          | 2.377             |
|        | MAPE     | 2.291%                                              | 2.290%                                          | **2.283%**      | 2.288%       | 2.331%                   | 2.304%        | 2.361%         | 2.447%            |
|        |          |                                                     |                                                 |                 |              |                          |               |                |                   |
| 30 min | MAE      | 1.338                                               | 1.338                                           | **1.336**       | 1.339        | 1.338                    | 1.346         | 1.419          | 1.395             |
|        | RMSE     | 2.900                                               | 2.902                                           | **2.898**       | 2.914        | 2.914                    | 2.937         | 3.147          | 3.008             |
|        | MAPE     | 2.844%                                              | 2.843%                                          | **2.836%**      | 2.853%       | 2.882%                   | 2.868%        | 3.049%         | 2.994%            |
|        |          |                                                     |                                                 |                 |              |                          |               |                |                   |
| 1 hour | MAE      | 1.577                                               | 1.577                                           | **1.576**       | 1.584        | 1.577                    | 1.584         | 1.765          | 1.637             |
|        | RMSE     | 3.578                                               | 3.577                                           | **3.569**       | 3.609        | 3.593                    | 3.601         | 4.069          | 3.676             |
|        | MAPE     | 3.524%                                              | **3.521%**                                      | 3.524%          | 3.562%       | 3.545%                   | 3.547%        | 4.067%         | 3.659%            |
