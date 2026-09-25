use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

/*
 * Load, clean, and do a whole lot more to csv files
 */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColumnType {
    Numeric,
    Boolean,
    Categorical,
    Empty,
}

impl ColumnType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Numeric => "numeric",
            Self::Boolean => "boolean",
            Self::Categorical => "categorical",
            Self::Empty => "empty",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnRole {
    Feature,
    Target,
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelTask {
    Regression,
    Classification,
    Clustering,
    DimensionalityReduction,
}

impl ModelTask {
    fn parse(value: &str) -> Result<Self, String> {
        let normalized = value.trim().to_ascii_lowercase().replace([' ', '-'], "_");

        match normalized.as_str() {
            "regression" => Ok(Self::Regression),
            "classification" => Ok(Self::Classification),
            "clustering" | "cluster" => Ok(Self::Clustering),
            "dimensionality_reduction" | "dim_reduction" | "pca" | "svd" => {
                Ok(Self::DimensionalityReduction)
            }
            _ => Err(format!(
                "Unknown task '{value}'. Use regression, classification, clustering, or dimensionality_reduction."
            )),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
            Self::DimensionalityReduction => "dimensionality_reduction",
        }
    }

    fn is_supervised(self) -> bool {
        matches!(self, Self::Regression | Self::Classification)
    }
}

pub(crate) struct FeatureEncoding {
    pub(crate) column_index: usize,
    pub(crate) column_name: String,
    pub(crate) kind: ColumnType,
    pub(crate) categories: Vec<String>,
    pub(crate) feature_start: usize,
    pub(crate) feature_names: Vec<String>,
}

pub(crate) enum ModelTarget {
    Numeric(Vec<f64>),
    Classes {
        class_ids: Vec<u32>,
        labels: Vec<String>,
    },
    None,
}

pub(crate) struct ModelInputs {
    pub(crate) task: ModelTask,
    pub(crate) x: Vec<Vec<f64>>,
    pub(crate) target: ModelTarget,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_encodings: Vec<FeatureEncoding>,
    pub(crate) feature_indices: Vec<usize>,
    pub(crate) target_index: Option<usize>,
    pub(crate) target_name: Option<String>,
    pub(crate) row_numbers: Vec<u32>,
}

impl ColumnRole {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "feature" => Ok(Self::Feature),
            "target" => Ok(Self::Target),
            "ignore" => Ok(Self::Ignore),
            _ => Err(format!(
                "Unknown column role '{value}'. Use feature, target, or ignore."
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Feature => "feature",
            Self::Target => "target",
            Self::Ignore => "ignore",
        }
    }
}

pub(crate) struct ParsedDataset {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    column_types: Vec<ColumnType>,
    missing_counts: Vec<usize>,
    distinct_counts: Vec<usize>,
    roles: Vec<ColumnRole>,
    row_start: u32,
    row_end: u32,
    delimiter: String,
    has_headers: bool,
}

impl ParsedDataset {
    pub(crate) fn from_csv(
        csv_text: &str,
        delimiter: &str,
        has_headers: bool,
    ) -> Result<Self, String> {
        let csv_text = csv_text.strip_prefix('\u{feff}').unwrap_or(csv_text);
        if csv_text.trim().is_empty() {
            return Err("The selected file is empty.".to_owned());
        }

        let delimiter_byte = resolve_delimiter(csv_text, delimiter)?;
        let delimiter_label = match delimiter_byte {
            b'\t' => "tab".to_owned(),
            value => char::from(value).to_string(),
        };

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .flexible(true)
            .delimiter(delimiter_byte)
            .from_reader(csv_text.as_bytes());

        let mut headers = if has_headers {
            reader
                .headers()
                .map_err(|error| format!("Could not read the header row: {error}"))?
                .iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut rows = Vec::new();
        for record in reader.records() {
            let record = record.map_err(|error| format!("Could not read a CSV row: {error}"))?;
            rows.push(record.iter().map(str::to_owned).collect::<Vec<_>>());
        }

        let row_width = rows.iter().map(Vec::len).max().unwrap_or(0);
        let column_count = headers.len().max(row_width);
        if column_count == 0 {
            return Err("The file has no columns.".to_owned());
        }

        if has_headers {
            while headers.len() < column_count {
                headers.push(format!("column_{}", headers.len() + 1));
            }
            for (index, header) in headers.iter_mut().enumerate() {
                if header.trim().is_empty() {
                    *header = format!("column_{}", index + 1);
                }
            }
        } else {
            headers = (1..=column_count)
                .map(|index| format!("column_{index}"))
                .collect();
        }

        // Keep ragged CSVs usable in the preview: short records get empty cells,
        // and extra cells get generated column names above.
        for row in &mut rows {
            row.resize(column_count, String::new());
        }

        let mut column_types = Vec::with_capacity(column_count);
        let mut missing_counts = Vec::with_capacity(column_count);
        let mut distinct_counts = Vec::with_capacity(column_count);
        for column in 0..column_count {
            let (kind, missing, distinct) = infer_column(&rows, column);
            column_types.push(kind);
            missing_counts.push(missing);
            distinct_counts.push(distinct);
        }

        let row_count = u32::try_from(rows.len())
            .map_err(|_| "The dataset has too many rows for the row selector.".to_owned())?;

        Ok(Self {
            headers,
            rows,
            column_types,
            missing_counts,
            distinct_counts,
            roles: vec![ColumnRole::Feature; column_count],
            row_start: if row_count == 0 { 0 } else { 1 },
            row_end: row_count,
            delimiter: delimiter_label,
            has_headers,
        })
    }

    pub(crate) fn snapshot_json(&self) -> Result<String, String> {
        let columns = self.column_metadata();
        let row_count = self.selected_row_count();
        let snapshot = json!({
            "headers": &self.headers,
            "rows": &self.rows,
            "columns": columns,
            "row_count": self.rows.len(),
            "column_count": self.headers.len(),
            "delimiter": self.delimiter,
            "has_headers": self.has_headers,
            "selected_range": {
                "start": self.row_start,
                "end": self.row_end,
                "row_count": row_count,
            },
        });

        serde_json::to_string(&snapshot).map_err(|error| error.to_string())
    }

    pub(crate) fn set_column_role(&mut self, column: u32, role: &str) -> Result<(), String> {
        let column =
            usize::try_from(column).map_err(|_| "Column index is out of range.".to_owned())?;
        if column >= self.headers.len() {
            return Err(format!("Column index {} is out of range.", column + 1));
        }

        let role = ColumnRole::parse(role)?;
        if role == ColumnRole::Target {
            for (index, current_role) in self.roles.iter_mut().enumerate() {
                if index != column && *current_role == ColumnRole::Target {
                    *current_role = ColumnRole::Feature;
                }
            }
        }
        self.roles[column] = role;
        Ok(())
    }

    pub(crate) fn column_roles_json(&self) -> Result<String, String> {
        let roles = self
            .roles
            .iter()
            .map(|role| role.as_str())
            .collect::<Vec<_>>();
        serde_json::to_string(&roles).map_err(|error| error.to_string())
    }

    pub(crate) fn set_row_range(&mut self, start: u32, end: u32) -> Result<(), String> {
        if self.rows.is_empty() {
            return Err("This dataset has no data rows to select.".to_owned());
        }
        if start == 0 || end < start || end as usize > self.rows.len() {
            return Err(format!(
                "Choose a row range from 1 to {} with the end row not before the start row.",
                self.rows.len()
            ));
        }

        self.row_start = start;
        self.row_end = end;
        Ok(())
    }

    pub(crate) fn selection_json(&self) -> Result<String, String> {
        let selected_rows: &[Vec<String>] = if self.row_start == 0 {
            &[]
        } else {
            &self.rows[(self.row_start as usize - 1)..self.row_end as usize]
        };
        let feature_indices = self
            .roles
            .iter()
            .enumerate()
            .filter_map(|(index, role)| (*role == ColumnRole::Feature).then_some(index))
            .collect::<Vec<_>>();
        let target_index = self
            .roles
            .iter()
            .position(|role| *role == ColumnRole::Target);

        let selection = json!({
            "headers": &self.headers,
            "columns": self.column_metadata(),
            "rows": selected_rows,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "selected_row_count": selected_rows.len(),
            "feature_indices": feature_indices,
            "target_index": target_index,
        });

        serde_json::to_string(&selection).map_err(|error| error.to_string())
    }

    pub(crate) fn model_inputs(&self, requested_task: &str) -> Result<ModelInputs, String> {
        let task = ModelTask::parse(requested_task)?;
        if self.row_start == 0 {
            return Err("There are no data rows in the selected range.".to_owned());
        }

        let row_start = self.row_start as usize - 1;
        let row_end = self.row_end as usize;
        let selected_rows = &self.rows[row_start..row_end];
        let feature_indices = self
            .roles
            .iter()
            .enumerate()
            .filter_map(|(index, role)| (*role == ColumnRole::Feature).then_some(index))
            .collect::<Vec<_>>();

        if feature_indices.is_empty() {
            return Err("Mark at least one column as a feature.".to_owned());
        }

        let target_index = if task.is_supervised() {
            Some(
                self.roles
                    .iter()
                    .position(|role| *role == ColumnRole::Target)
                    .ok_or_else(|| {
                        format!("Mark one column as the target for {}.", task.as_str())
                    })?,
            )
        } else {
            None
        };

        let mut encodings = Vec::with_capacity(feature_indices.len());
        let mut feature_names = Vec::new();

        for column_index in feature_indices.iter().copied() {
            let kind = self.column_types[column_index];
            if kind == ColumnType::Empty {
                return Err(format!(
                    "Feature column '{}' has no values in the dataset.",
                    self.headers[column_index]
                ));
            }

            let mut categories = Vec::new();
            if kind == ColumnType::Categorical {
                let mut seen_categories = HashSet::new();
                for (row_offset, row) in selected_rows.iter().enumerate() {
                    let row_number = self.row_start as usize + row_offset;
                    let value =
                        nonblank_cell(&row[column_index], row_number, &self.headers[column_index])?;
                    let category = value.to_owned();
                    if seen_categories.insert(category.clone()) {
                        categories.push(category);
                    }
                }
            }

            let names = if kind == ColumnType::Categorical {
                categories
                    .iter()
                    .map(|category| format!("{}={category}", self.headers[column_index]))
                    .collect::<Vec<_>>()
            } else {
                vec![self.headers[column_index].clone()]
            };
            let feature_start = feature_names.len();
            feature_names.extend(names.iter().cloned());
            encodings.push(FeatureEncoding {
                column_index,
                column_name: self.headers[column_index].clone(),
                kind,
                categories,
                feature_start,
                feature_names: names,
            });
        }

        let mut x = vec![vec![0.0; feature_names.len()]; selected_rows.len()];
        for (row_offset, row) in selected_rows.iter().enumerate() {
            let row_number = self.row_start as usize + row_offset;
            for encoding in &encodings {
                let value = nonblank_cell(
                    &row[encoding.column_index],
                    row_number,
                    &self.headers[encoding.column_index],
                )?;

                match encoding.kind {
                    ColumnType::Numeric => {
                        x[row_offset][encoding.feature_start] =
                            numeric_cell(value, row_number, &self.headers[encoding.column_index])?;
                    }
                    ColumnType::Boolean => {
                        x[row_offset][encoding.feature_start] =
                            boolean_cell(value, row_number, &self.headers[encoding.column_index])?
                                as u8 as f64;
                    }
                    ColumnType::Categorical => {
                        let category_index = encoding
                            .categories
                            .iter()
                            .position(|category| category == value)
                            .ok_or_else(|| {
                                format!(
                                    "Unknown category '{value}' in row {row_number}, column '{}'.",
                                    self.headers[encoding.column_index]
                                )
                            })?;
                        x[row_offset][encoding.feature_start + category_index] = 1.0;
                    }
                    ColumnType::Empty => unreachable!("empty columns are rejected above"),
                }
            }
        }

        let target = match task {
            ModelTask::Regression => {
                let target_index = target_index.expect("supervised tasks have a target");
                let target_name = &self.headers[target_index];
                let target_type = self.column_types[target_index];
                let values = selected_rows
                    .iter()
                    .enumerate()
                    .map(|(row_offset, row)| {
                        let row_number = self.row_start as usize + row_offset;
                        let value = nonblank_cell(&row[target_index], row_number, target_name)?;
                        if target_type == ColumnType::Boolean {
                            Ok(boolean_cell(value, row_number, target_name)? as u8 as f64)
                        } else {
                            numeric_cell(value, row_number, target_name)
                        }
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                ModelTarget::Numeric(values)
            }
            ModelTask::Classification => {
                let target_index = target_index.expect("supervised tasks have a target");
                let target_name = &self.headers[target_index];
                let mut class_to_id = HashMap::<String, u32>::new();
                let mut labels = Vec::new();
                let mut values = Vec::with_capacity(selected_rows.len());

                for (row_offset, row) in selected_rows.iter().enumerate() {
                    let row_number = self.row_start as usize + row_offset;
                    let label = nonblank_cell(&row[target_index], row_number, target_name)?;
                    let class_id = match class_to_id.get(label) {
                        Some(class_id) => *class_id,
                        None => {
                            let class_id = u32::try_from(labels.len())
                                .map_err(|_| "The target has too many classes.".to_owned())?;
                            let label = label.to_owned();
                            class_to_id.insert(label.clone(), class_id);
                            labels.push(label);
                            class_id
                        }
                    };
                    values.push(class_id);
                }

                ModelTarget::Classes {
                    class_ids: values,
                    labels,
                }
            }
            ModelTask::Clustering | ModelTask::DimensionalityReduction => ModelTarget::None,
        };

        let row_numbers = (self.row_start..=self.row_end).collect::<Vec<_>>();

        Ok(ModelInputs {
            task,
            x,
            target,
            feature_names,
            feature_encodings: encodings,
            feature_indices,
            target_index,
            target_name: target_index.map(|index| self.headers[index].clone()),
            row_numbers,
        })
    }

    pub(crate) fn model_input_json(&self, requested_task: &str) -> Result<String, String> {
        self.model_inputs(requested_task)?.to_json()
    }

    fn selected_row_count(&self) -> u32 {
        if self.row_start == 0 {
            0
        } else {
            self.row_end - self.row_start + 1
        }
    }

    fn column_metadata(&self) -> Vec<Value> {
        (0..self.headers.len())
            .map(|index| {
                json!({
                    "index": index,
                    "name": &self.headers[index],
                    "type": self.column_types[index].as_str(),
                    "role": self.roles[index].as_str(),
                    "missing_count": self.missing_counts[index],
                    "distinct_count": self.distinct_counts[index],
                })
            })
            .collect()
    }
}

impl ModelInputs {
    pub(crate) fn to_json(&self) -> Result<String, String> {
        let (y_kind, y, class_labels) = match &self.target {
            ModelTarget::Numeric(values) => ("numeric", json!(values), Value::Null),
            ModelTarget::Classes { class_ids, labels } => {
                ("class", json!(class_ids), json!(labels))
            }
            ModelTarget::None => ("none", Value::Null, Value::Null),
        };

        let feature_encodings = self
            .feature_encodings
            .iter()
            .map(|encoding| {
                json!({
                    "source_column_index": encoding.column_index,
                    "source_column_name": &encoding.column_name,
                    "source_type": encoding.kind.as_str(),
                    "category_values": if encoding.kind == ColumnType::Categorical {
                        json!(&encoding.categories)
                    } else {
                        Value::Null
                    },
                    "feature_start": encoding.feature_start,
                    "feature_count": encoding.feature_names.len(),
                    "feature_names": &encoding.feature_names,
                })
            })
            .collect::<Vec<_>>();

        let model_input = json!({
            "task": self.task.as_str(),
            "x": &self.x,
            "y": y,
            "y_kind": y_kind,
            "class_labels": class_labels,
            "feature_names": &self.feature_names,
            "feature_encodings": feature_encodings,
            "feature_indices": &self.feature_indices,
            "target_index": self.target_index,
            "target_name": &self.target_name,
            "row_numbers": &self.row_numbers,
            "row_start": self.row_numbers.first().copied().unwrap_or(0),
            "row_end": self.row_numbers.last().copied().unwrap_or(0),
            "row_count": self.x.len(),
            "feature_count": self.feature_names.len(),
        });

        serde_json::to_string(&model_input).map_err(|error| error.to_string())
    }
}

fn infer_column(rows: &[Vec<String>], column: usize) -> (ColumnType, usize, usize) {
    let mut has_value = false;
    let mut all_numeric = true;
    let mut all_boolean = true;
    let mut missing_count = 0;
    let mut distinct_values = HashSet::new();

    for row in rows {
        let value = row[column].trim();
        if value.is_empty() {
            missing_count += 1;
            continue;
        }

        has_value = true;
        distinct_values.insert(value);
        if !value
            .parse::<f64>()
            .map(|number| number.is_finite())
            .unwrap_or(false)
        {
            all_numeric = false;
        }
        if !matches!(value.to_ascii_lowercase().as_str(), "true" | "false") {
            all_boolean = false;
        }
    }

    let kind = if !has_value {
        ColumnType::Empty
    } else if all_numeric {
        ColumnType::Numeric
    } else if all_boolean {
        ColumnType::Boolean
    } else {
        ColumnType::Categorical
    };

    (kind, missing_count, distinct_values.len())
}

fn nonblank_cell<'a>(
    value: &'a str,
    row_number: usize,
    column_name: &str,
) -> Result<&'a str, String> {
    let value = value.trim();
    if value.is_empty() {
        Err(format!(
            "Blank value at data row {row_number}, column '{column_name}'. Missing values are not imputed."
        ))
    } else {
        Ok(value)
    }
}

fn numeric_cell(value: &str, row_number: usize, column_name: &str) -> Result<f64, String> {
    let number = value.parse::<f64>().map_err(|_| {
        format!("Value '{value}' at data row {row_number}, column '{column_name}' is not numeric.")
    })?;
    if !number.is_finite() {
        return Err(format!(
            "Value '{value}' at data row {row_number}, column '{column_name}' is not a finite number."
        ));
    }
    Ok(number)
}

fn boolean_cell(value: &str, row_number: usize, column_name: &str) -> Result<bool, String> {
    match value.to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!(
            "Value '{value}' at data row {row_number}, column '{column_name}' is not true or false."
        )),
    }
}

fn resolve_delimiter(csv_text: &str, requested: &str) -> Result<u8, String> {
    match requested {
        "auto" => Ok(detect_delimiter(csv_text)),
        "," | "comma" => Ok(b','),
        ";" | "semicolon" => Ok(b';'),
        "\t" | "tab" => Ok(b'\t'),
        _ => Err("Delimiter must be auto, comma, semicolon, or tab.".to_owned()),
    }
}

fn detect_delimiter(csv_text: &str) -> u8 {
    let first_line = csv_text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    let counts = count_delimiters(first_line);
    let candidates = [b',', b';', b'\t'];

    let (index, count) = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(index, count)| (index, *count))
        .unwrap_or((0, 0));

    if count == 0 { b',' } else { candidates[index] }
}

fn count_delimiters(line: &str) -> [usize; 3] {
    let characters = line.chars().collect::<Vec<_>>();
    let mut counts = [0; 3];
    let mut quoted = false;
    let mut index = 0;

    while index < characters.len() {
        match characters[index] {
            '"' if quoted && characters.get(index + 1) == Some(&'"') => index += 1,
            '"' => quoted = !quoted,
            ',' if !quoted => counts[0] += 1,
            ';' if !quoted => counts[1] += 1,
            '\t' if !quoted => counts[2] += 1,
            _ => {}
        }
        index += 1;
    }

    counts
}
