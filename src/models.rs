use std::collections::BTreeMap;

use serde_json::{Value, json};
use smartcore::cluster::agglomerative::{
    AgglomerativeClustering, AgglomerativeClusteringParameters,
};
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::decomposition::pca::{PCA, PCAParameters};
use smartcore::decomposition::svd::{SVD, SVDParameters};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};
use smartcore::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};

use crate::dataset::{ModelInputs, ModelTarget, ModelTask};

#[derive(Debug)]
struct RunOptions {
    test_percent: usize,
    seed: u64,
    n_trees: usize,
    max_depth: usize,
    k: usize,
    n_components: usize,
    max_iter: usize,
    epochs: usize,
    learning_rate: f64,
}

impl RunOptions {
    fn parse(options_json: &str) -> Result<Self, String> {
        let value: Value = if options_json.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(options_json)
                .map_err(|error| format!("Invalid model options: {error}"))?
        };

        let usize_option = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(default)
        };
        let seed = value.get("seed").and_then(Value::as_u64).unwrap_or(42);
        let test_percent = usize_option("test_percent", 25).clamp(5, 60);
        let learning_rate = value
            .get("learning_rate")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(0.1);

        Ok(Self {
            test_percent,
            seed,
            n_trees: usize_option("n_trees", 100).clamp(1, u16::MAX as usize),
            max_depth: usize_option("max_depth", 0),
            k: usize_option("k", 3).max(1),
            n_components: usize_option("n_components", 2).max(1),
            max_iter: usize_option("max_iter", 100).max(1),
            epochs: usize_option("epochs", 100).max(1),
            learning_rate,
        })
    }
}

pub(crate) fn run_model(
    inputs: ModelInputs,
    algorithm: &str,
    options_json: &str,
) -> Result<String, String> {
    let options = RunOptions::parse(options_json)?;
    let result = match inputs.task {
        ModelTask::Regression => run_regression(inputs, algorithm, &options)?,
        ModelTask::Classification => run_classification(inputs, algorithm, &options)?,
        ModelTask::Clustering => run_clustering(inputs, algorithm, &options)?,
        ModelTask::DimensionalityReduction => {
            run_dimensionality_reduction(inputs, algorithm, &options)?
        }
    };

    serde_json::to_string(&result).map_err(|error| error.to_string())
}

fn run_regression(
    inputs: ModelInputs,
    algorithm: &str,
    options: &RunOptions,
) -> Result<Value, String> {
    let ModelInputs {
        x,
        target,
        feature_names,
        row_numbers,
        ..
    } = inputs;
    let ModelTarget::Numeric(y) = target else {
        return Err("Regression requires a numeric or boolean target column.".to_owned());
    };
    if y.len() != x.len() {
        return Err("Feature and target row counts do not match.".to_owned());
    }
    if x.len() < 2 {
        return Err("Regression needs at least two selected rows.".to_owned());
    }

    let (train_indices, test_indices) = split_indices(x.len(), options.test_percent, options.seed)?;
    let mut train_x = select_rows(&x, &train_indices);
    let mut test_x = select_rows(&x, &test_indices);
    let train_y = select_values(&y, &train_indices);
    let actual = select_values(&y, &test_indices);

    let predicted = match algorithm {
        "linear_regression" => {
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let model = LinearRegression::fit(
                &train_matrix,
                &train_y,
                LinearRegressionParameters::default(),
            )
            .map_err(|error| format!("Linear regression could not fit: {error:?}"))?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("Linear regression prediction failed: {error:?}"))?
        }
        "decision_tree_regressor" => {
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let mut parameters = DecisionTreeRegressorParameters::default();
            if options.max_depth > 0 {
                parameters =
                    parameters.with_max_depth(options.max_depth.min(u16::MAX as usize) as u16);
            }
            parameters.seed = Some(options.seed);
            let model = DecisionTreeRegressor::fit(&train_matrix, &train_y, parameters)
                .map_err(|error| format!("Decision-tree regression could not fit: {error:?}"))?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("Decision-tree prediction failed: {error:?}"))?
        }
        "random_forest_regressor" => {
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let parameters = RandomForestRegressorParameters::default()
                .with_n_trees(options.n_trees)
                .with_seed(options.seed);
            let parameters = if options.max_depth > 0 {
                parameters.with_max_depth(options.max_depth.min(u16::MAX as usize) as u16)
            } else {
                parameters
            };
            let model = RandomForestRegressor::fit(&train_matrix, &train_y, parameters)
                .map_err(|error| format!("Random-forest regression could not fit: {error:?}"))?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("Random-forest prediction failed: {error:?}"))?
        }
        "knn_regressor" => {
            standardize_pair(&mut train_x, &mut test_x);
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let k = options.k.min(train_x.len()).max(1);
            let parameters = KNNRegressorParameters::<f64, Euclidian<f64>>::default().with_k(k);
            let model = KNNRegressor::fit(&train_matrix, &train_y, parameters)
                .map_err(|error| format!("KNN regression could not fit: {error:?}"))?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("KNN prediction failed: {error:?}"))?
        }
        "poisson_glm" => poisson_regression(&train_x, &train_y, &test_x, options.max_iter)?,
        _ => {
            return Err(format!("Unknown regression algorithm '{algorithm}'."));
        }
    };

    ensure_finite(&predicted, "Model predictions")?;
    let test_rows = select_values(&row_numbers, &test_indices);
    let mut metrics = regression_metrics(&actual, &predicted);
    if algorithm == "poisson_glm" {
        metrics["poisson_deviance"] = json!(poisson_deviance(&actual, &predicted));
    }

    Ok(json!({
        "task": "regression",
        "algorithm": algorithm,
        "metrics": metrics,
        "actual": actual,
        "predicted": predicted,
        "test_row_numbers": test_rows,
        "feature_names": feature_names,
        "train_rows": train_indices.len(),
        "test_rows": test_indices.len(),
    }))
}

fn run_classification(
    inputs: ModelInputs,
    algorithm: &str,
    options: &RunOptions,
) -> Result<Value, String> {
    let ModelInputs {
        x,
        target,
        feature_names,
        row_numbers,
        ..
    } = inputs;
    let ModelTarget::Classes { class_ids, labels } = target else {
        return Err("Classification requires a target column.".to_owned());
    };
    if class_ids.len() != x.len() {
        return Err("Feature and target row counts do not match.".to_owned());
    }
    if labels.len() < 2 {
        return Err("Classification needs at least two distinct target classes.".to_owned());
    }
    if x.len() < 2 {
        return Err("Classification needs at least two selected rows.".to_owned());
    }

    let (train_indices, test_indices) =
        stratified_split(&class_ids, options.test_percent, options.seed)?;
    let mut train_x = select_rows(&x, &train_indices);
    let mut test_x = select_rows(&x, &test_indices);
    let train_y = select_values(&class_ids, &train_indices);
    let actual = select_values(&class_ids, &test_indices);

    let predicted = match algorithm {
        "logistic_regression" => {
            standardize_pair(&mut train_x, &mut test_x);
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let model = LogisticRegression::fit(
                &train_matrix,
                &train_y,
                LogisticRegressionParameters::<f64>::default(),
            )
            .map_err(|error| format!("Logistic regression could not fit: {error:?}"))?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("Logistic-regression prediction failed: {error:?}"))?
        }
        "decision_tree_classifier" => {
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let mut parameters = DecisionTreeClassifierParameters::default();
            if options.max_depth > 0 {
                parameters =
                    parameters.with_max_depth(options.max_depth.min(u16::MAX as usize) as u16);
            }
            parameters.seed = Some(options.seed);
            let model = DecisionTreeClassifier::fit(&train_matrix, &train_y, parameters).map_err(
                |error| format!("Decision-tree classification could not fit: {error:?}"),
            )?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("Decision-tree prediction failed: {error:?}"))?
        }
        "random_forest_classifier" => {
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let tree_count = options.n_trees.min(u16::MAX as usize) as u16;
            let mut parameters = RandomForestClassifierParameters::default()
                .with_n_trees(tree_count)
                .with_seed(options.seed);
            if options.max_depth > 0 {
                parameters =
                    parameters.with_max_depth(options.max_depth.min(u16::MAX as usize) as u16);
            }
            let model = RandomForestClassifier::fit(&train_matrix, &train_y, parameters).map_err(
                |error| format!("Random-forest classification could not fit: {error:?}"),
            )?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("Random-forest prediction failed: {error:?}"))?
        }
        "knn_classifier" => {
            standardize_pair(&mut train_x, &mut test_x);
            let train_matrix = dense_matrix(&train_x)?;
            let test_matrix = dense_matrix(&test_x)?;
            let k = options.k.min(train_x.len()).max(1);
            let parameters = KNNClassifierParameters::<f64, Euclidian<f64>>::default().with_k(k);
            let model = KNNClassifier::fit(&train_matrix, &train_y, parameters)
                .map_err(|error| format!("KNN classification could not fit: {error:?}"))?;
            model
                .predict(&test_matrix)
                .map_err(|error| format!("KNN prediction failed: {error:?}"))?
        }
        "perceptron" => perceptron(
            &train_x,
            &train_y,
            &test_x,
            labels.len(),
            options.epochs,
            options.learning_rate,
            options.seed,
        )?,
        _ => {
            return Err(format!("Unknown classification algorithm '{algorithm}'."));
        }
    };

    let test_rows = select_values(&row_numbers, &test_indices);
    let (metrics, confusion_matrix) = classification_metrics(&actual, &predicted, labels.len());
    let actual_labels = actual
        .iter()
        .map(|&id| labels[id as usize].clone())
        .collect::<Vec<_>>();
    let predicted_labels = predicted
        .iter()
        .map(|&id| {
            labels
                .get(id as usize)
                .cloned()
                .unwrap_or_else(|| format!("class {id}"))
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "task": "classification",
        "algorithm": algorithm,
        "metrics": metrics,
        "class_labels": labels,
        "actual_class_ids": actual,
        "predicted_class_ids": predicted,
        "actual_labels": actual_labels,
        "predicted_labels": predicted_labels,
        "confusion_matrix": confusion_matrix,
        "test_row_numbers": test_rows,
        "feature_names": feature_names,
        "train_rows": train_indices.len(),
        "test_rows": test_indices.len(),
    }))
}

fn run_clustering(
    inputs: ModelInputs,
    algorithm: &str,
    options: &RunOptions,
) -> Result<Value, String> {
    let row_numbers = inputs.row_numbers;
    let feature_names = inputs.feature_names;
    let mut x = inputs.x;
    if x.len() < 2 {
        return Err("Clustering needs at least two selected rows.".to_owned());
    }
    standardize_all(&mut x);
    let matrix = dense_matrix(&x)?;
    let cluster_count = options.k.clamp(2, x.len());

    let labels = match algorithm {
        "kmeans" => {
            let parameters = KMeansParameters::default()
                .with_k(cluster_count)
                .with_max_iter(options.max_iter);
            let parameters = KMeansParameters {
                seed: Some(options.seed),
                ..parameters
            };
            let model = KMeans::<f64, u32, DenseMatrix<f64>, Vec<u32>>::fit(&matrix, parameters)
                .map_err(|error| format!("K-means could not fit: {error:?}"))?;
            model
                .predict(&matrix)
                .map_err(|error| format!("K-means prediction failed: {error:?}"))?
                .into_iter()
                .map(|label| label as usize)
                .collect::<Vec<_>>()
        }
        "hierarchical" => {
            let parameters =
                AgglomerativeClusteringParameters::default().with_n_clusters(cluster_count);
            AgglomerativeClustering::<f64, usize, DenseMatrix<f64>, Vec<usize>>::fit(
                &matrix, parameters,
            )
            .map_err(|error| format!("Hierarchical clustering failed: {error:?}"))?
            .labels
        }
        _ => return Err(format!("Unknown clustering algorithm '{algorithm}'.")),
    };

    let cluster_sizes = cluster_sizes(&labels);
    Ok(json!({
        "task": "clustering",
        "algorithm": algorithm,
        "cluster_labels": labels,
        "cluster_sizes": cluster_sizes,
        "coordinates": plot_coordinates(&x),
        "row_numbers": row_numbers,
        "feature_names": feature_names,
    }))
}

fn run_dimensionality_reduction(
    inputs: ModelInputs,
    algorithm: &str,
    options: &RunOptions,
) -> Result<Value, String> {
    let row_numbers = inputs.row_numbers;
    let feature_names = inputs.feature_names;
    let mut x = inputs.x;
    if x.is_empty() {
        return Err("Dimensionality reduction needs selected rows.".to_owned());
    }
    let feature_count = x[0].len();
    if feature_count == 0 {
        return Err("Dimensionality reduction needs at least one feature.".to_owned());
    }
    standardize_all(&mut x);
    let matrix = dense_matrix(&x)?;

    let (embedding, component_count) = match algorithm {
        "pca" => {
            let count = options.n_components.min(feature_count).max(1);
            let parameters = PCAParameters::default()
                .with_n_components(count)
                .with_use_correlation_matrix(false);
            let model = PCA::<f64, DenseMatrix<f64>>::fit(&matrix, parameters)
                .map_err(|error| format!("PCA could not fit: {error:?}"))?;
            let transformed = model
                .transform(&matrix)
                .map_err(|error| format!("PCA transform failed: {error:?}"))?;
            (matrix_to_rows(&transformed), count)
        }
        "svd" => {
            if feature_count < 2 {
                return Err("SVD needs at least two input features.".to_owned());
            }
            let count = options.n_components.min(feature_count - 1).max(1);
            let model = SVD::<f64, DenseMatrix<f64>>::fit(
                &matrix,
                SVDParameters::default().with_n_components(count),
            )
            .map_err(|error| format!("SVD could not fit: {error:?}"))?;
            let transformed = model
                .transform(&matrix)
                .map_err(|error| format!("SVD transform failed: {error:?}"))?;
            (matrix_to_rows(&transformed), count)
        }
        _ => {
            return Err(format!(
                "Unknown dimensionality-reduction algorithm '{algorithm}'."
            ));
        }
    };

    Ok(json!({
        "task": "dimensionality_reduction",
        "algorithm": algorithm,
        "component_count": component_count,
        "component_names": (1..=component_count).map(|index| format!("Component {index}")).collect::<Vec<_>>(),
        "embedding": embedding,
        "row_numbers": row_numbers,
        "input_feature_names": feature_names,
    }))
}

fn dense_matrix(rows: &[Vec<f64>]) -> Result<DenseMatrix<f64>, String> {
    DenseMatrix::from_2d_vec(&rows.to_vec())
        .map_err(|error| format!("Could not create matrix: {error:?}"))
}

fn matrix_to_rows(matrix: &DenseMatrix<f64>) -> Vec<Vec<f64>> {
    let (row_count, column_count) = matrix.shape();
    (0..row_count)
        .map(|row| {
            (0..column_count)
                .map(|column| *matrix.get((row, column)))
                .collect()
        })
        .collect()
}

fn select_rows(rows: &[Vec<f64>], indices: &[usize]) -> Vec<Vec<f64>> {
    indices.iter().map(|&index| rows[index].clone()).collect()
}

fn select_values<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&index| values[index]).collect()
}

fn split_indices(
    row_count: usize,
    test_percent: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if row_count < 2 {
        return Err("A train/test split needs at least two selected rows.".to_owned());
    }

    let test_count =
        ((row_count as f64 * test_percent as f64 / 100.0).round() as usize).clamp(1, row_count - 1);
    let mut indices = (0..row_count).collect::<Vec<_>>();
    let mut rng = StableRng::new(seed);
    shuffle(&mut indices, &mut rng);
    let mut test = indices.drain(..test_count).collect::<Vec<_>>();
    let mut train = indices;
    train.sort_unstable();
    test.sort_unstable();
    Ok((train, test))
}

fn stratified_split(
    labels: &[u32],
    test_percent: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let mut groups = BTreeMap::<u32, Vec<usize>>::new();
    for (index, &label) in labels.iter().enumerate() {
        groups.entry(label).or_default().push(index);
    }

    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut rng = StableRng::new(seed);
    for indices in groups.values_mut() {
        shuffle(indices, &mut rng);
        let test_count = if indices.len() < 2 {
            0
        } else {
            ((indices.len() as f64 * test_percent as f64 / 100.0).round() as usize)
                .clamp(1, indices.len() - 1)
        };
        test.extend(indices[..test_count].iter().copied());
        train.extend(indices[test_count..].iter().copied());
    }

    if train.is_empty() || test.is_empty() {
        return Err(
            "Classification needs enough rows to leave at least one row in both train and test sets."
                .to_owned(),
        );
    }

    train.sort_unstable();
    test.sort_unstable();
    Ok((train, test))
}

fn shuffle<T>(values: &mut [T], rng: &mut StableRng) {
    for index in (1..values.len()).rev() {
        let other = (rng.next_u64() % (index as u64 + 1)) as usize;
        values.swap(index, other);
    }
}

struct StableRng {
    state: u64,
}

impl StableRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn scaling_parameters(rows: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let feature_count = rows.first().map_or(0, Vec::len);
    let mut means = vec![0.0; feature_count];
    for row in rows {
        for (column, value) in row.iter().enumerate() {
            means[column] += value;
        }
    }
    for mean in &mut means {
        *mean /= rows.len() as f64;
    }

    let mut scales = vec![0.0; feature_count];
    for row in rows {
        for (column, value) in row.iter().enumerate() {
            scales[column] += (value - means[column]).powi(2);
        }
    }
    for scale in &mut scales {
        *scale = (*scale / rows.len() as f64).sqrt();
        if *scale < 1e-12 {
            *scale = 1.0;
        }
    }
    (means, scales)
}

fn apply_scaling(rows: &mut [Vec<f64>], means: &[f64], scales: &[f64]) {
    for row in rows {
        for column in 0..row.len() {
            row[column] = (row[column] - means[column]) / scales[column];
        }
    }
}

fn standardize_pair(train: &mut [Vec<f64>], other: &mut [Vec<f64>]) {
    let (means, scales) = scaling_parameters(train);
    apply_scaling(train, &means, &scales);
    apply_scaling(other, &means, &scales);
}

fn standardize_all(rows: &mut [Vec<f64>]) {
    let (means, scales) = scaling_parameters(rows);
    apply_scaling(rows, &means, &scales);
}

fn regression_metrics(actual: &[f64], predicted: &[f64]) -> Value {
    let count = actual.len().max(1) as f64;
    let mean = actual.iter().sum::<f64>() / count;
    let mae = actual
        .iter()
        .zip(predicted)
        .map(|(actual, predicted)| (actual - predicted).abs())
        .sum::<f64>()
        / count;
    let squared_error = actual
        .iter()
        .zip(predicted)
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>();
    let total_sum_squares = actual
        .iter()
        .map(|actual| (actual - mean).powi(2))
        .sum::<f64>();
    let r2 = if total_sum_squares < 1e-12 {
        None
    } else {
        Some(1.0 - squared_error / total_sum_squares)
    };

    json!({
        "mae": mae,
        "rmse": (squared_error / count).sqrt(),
        "r2": r2,
    })
}

fn poisson_deviance(actual: &[f64], predicted: &[f64]) -> f64 {
    2.0 * actual
        .iter()
        .zip(predicted)
        .map(|(&observed, &mean)| {
            if observed <= 0.0 {
                mean
            } else {
                observed * (observed / mean.max(1e-12)).ln() - (observed - mean)
            }
        })
        .sum::<f64>()
}

fn classification_metrics(
    actual: &[u32],
    predicted: &[u32],
    class_count: usize,
) -> (Value, Vec<Vec<usize>>) {
    let mut confusion = vec![vec![0usize; class_count]; class_count];
    for (&actual, &predicted) in actual.iter().zip(predicted) {
        let actual = actual as usize;
        let predicted = predicted as usize;
        if actual < class_count && predicted < class_count {
            confusion[actual][predicted] += 1;
        }
    }

    let correct = (0..class_count)
        .map(|class| confusion[class][class])
        .sum::<usize>();
    let accuracy = correct as f64 / actual.len().max(1) as f64;
    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut f1_sum = 0.0;

    for class in 0..class_count {
        let true_positive = confusion[class][class] as f64;
        let predicted_count = (0..class_count)
            .map(|row| confusion[row][class])
            .sum::<usize>() as f64;
        let actual_count = confusion[class].iter().sum::<usize>() as f64;
        let precision = if predicted_count == 0.0 {
            0.0
        } else {
            true_positive / predicted_count
        };
        let recall = if actual_count == 0.0 {
            0.0
        } else {
            true_positive / actual_count
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
    }

    let divisor = class_count.max(1) as f64;
    (
        json!({
            "accuracy": accuracy,
            "precision_macro": precision_sum / divisor,
            "recall_macro": recall_sum / divisor,
            "f1_macro": f1_sum / divisor,
        }),
        confusion,
    )
}

fn cluster_sizes(labels: &[usize]) -> Vec<usize> {
    let max_label = labels.iter().copied().max().unwrap_or(0);
    let mut sizes = vec![0usize; max_label + 1];
    for &label in labels {
        sizes[label] += 1;
    }
    sizes
}

fn plot_coordinates(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            vec![
                row.first().copied().unwrap_or(0.0),
                if row.len() > 1 { row[1] } else { 0.0 },
            ]
        })
        .collect()
}

fn ensure_finite(values: &[f64], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(format!("{label} contain a non-finite number."))
    } else {
        Ok(())
    }
}

fn poisson_regression(
    train_x: &[Vec<f64>],
    train_y: &[f64],
    test_x: &[Vec<f64>],
    max_iter: usize,
) -> Result<Vec<f64>, String> {
    if train_y.iter().any(|value| *value < 0.0) {
        return Err("Poisson GLM requires a non-negative target.".to_owned());
    }

    let mut train_x = train_x.to_vec();
    let mut test_x = test_x.to_vec();
    standardize_pair(&mut train_x, &mut test_x);
    let coefficient_count = train_x[0].len() + 1;
    let mut coefficients = vec![0.0; coefficient_count];

    for _ in 0..max_iter {
        let mut normal = vec![vec![0.0; coefficient_count]; coefficient_count];
        let mut response = vec![0.0; coefficient_count];

        for (row, &target) in train_x.iter().zip(train_y) {
            let mut design = Vec::with_capacity(coefficient_count);
            design.push(1.0);
            design.extend(row.iter().copied());
            let eta = dot(&coefficients, &design).clamp(-20.0, 20.0);
            let mean = eta.exp().max(1e-10);
            let adjusted = eta + (target - mean) / mean;

            for i in 0..coefficient_count {
                response[i] += mean * design[i] * adjusted;
                for j in 0..coefficient_count {
                    normal[i][j] += mean * design[i] * design[j];
                }
            }
        }

        for diagonal in 1..coefficient_count {
            normal[diagonal][diagonal] += 1e-8;
        }
        let next = solve_linear_system(normal, response)?;
        let change = next
            .iter()
            .zip(&coefficients)
            .map(|(next, current)| (next - current).abs())
            .fold(0.0, f64::max);
        coefficients = next;
        if change < 1e-7 {
            break;
        }
    }

    let predicted = test_x
        .iter()
        .map(|row| {
            let eta = coefficients[0] + dot(&coefficients[1..], row);
            eta.clamp(-20.0, 20.0).exp()
        })
        .collect::<Vec<_>>();
    ensure_finite(&predicted, "Poisson predictions")?;
    Ok(predicted)
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>, String> {
    let size = rhs.len();
    for pivot in 0..size {
        let pivot_row = (pivot..size)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .total_cmp(&matrix[right][pivot].abs())
            })
            .ok_or_else(|| "Could not solve the Poisson GLM system.".to_owned())?;
        if matrix[pivot_row][pivot].abs() < 1e-14 {
            return Err("Poisson GLM could not solve a singular feature matrix.".to_owned());
        }
        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);

        let divisor = matrix[pivot][pivot];
        for column in pivot..size {
            matrix[pivot][column] /= divisor;
        }
        rhs[pivot] /= divisor;

        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..size {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}

fn perceptron(
    train_x: &[Vec<f64>],
    train_y: &[u32],
    test_x: &[Vec<f64>],
    class_count: usize,
    epochs: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<Vec<u32>, String> {
    let mut train_x = train_x.to_vec();
    let mut test_x = test_x.to_vec();
    standardize_pair(&mut train_x, &mut test_x);
    let feature_count = train_x[0].len();
    let mut weights = vec![vec![0.0; feature_count + 1]; class_count];
    let mut order = (0..train_x.len()).collect::<Vec<_>>();
    let mut rng = StableRng::new(seed);

    for _ in 0..epochs {
        shuffle(&mut order, &mut rng);
        for &row_index in &order {
            let actual = train_y[row_index] as usize;
            if actual >= class_count {
                return Err("Classification target contains an invalid class id.".to_owned());
            }
            let predicted = perceptron_class(&weights, &train_x[row_index]);
            if predicted != actual {
                for column in 0..feature_count {
                    weights[actual][column] += learning_rate * train_x[row_index][column];
                    weights[predicted][column] -= learning_rate * train_x[row_index][column];
                }
                weights[actual][feature_count] += learning_rate;
                weights[predicted][feature_count] -= learning_rate;
            }
        }
    }

    Ok(test_x
        .iter()
        .map(|row| perceptron_class(&weights, row) as u32)
        .collect())
}

fn perceptron_class(weights: &[Vec<f64>], row: &[f64]) -> usize {
    weights
        .iter()
        .enumerate()
        .map(|(class, weight)| {
            let score = weight[..row.len()]
                .iter()
                .zip(row)
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + weight[row.len()];
            (class, score)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(class, _)| class)
        .unwrap_or(0)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
