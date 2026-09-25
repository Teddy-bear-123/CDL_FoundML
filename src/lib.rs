mod dataset;
mod models;

use dataset::{ModelInputs, ParsedDataset};
use wasm_bindgen::prelude::*;

/*
 * Cleaned up dataset structure to be used later
 */
#[wasm_bindgen]
pub struct Dataset {
    inner: ParsedDataset,
}

#[wasm_bindgen]
impl Dataset {
    #[wasm_bindgen(constructor)]
    pub fn new(csv_text: &str, delimiter: &str, has_headers: bool) -> Result<Dataset, JsValue> {
        let inner = ParsedDataset::from_csv(csv_text, delimiter, has_headers)
            .map_err(|error| JsValue::from_str(&error))?;

        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = snapshotJson)]
    pub fn snapshot_json(&self) -> Result<String, JsValue> {
        self.inner
            .snapshot_json()
            .map_err(|error| JsValue::from_str(&error))
    }

    #[wasm_bindgen(js_name = setColumnRole)]
    pub fn set_column_role(&mut self, column: u32, role: &str) -> Result<(), JsValue> {
        self.inner
            .set_column_role(column, role)
            .map_err(|error| JsValue::from_str(&error))
    }

    #[wasm_bindgen(js_name = columnRolesJson)]
    pub fn column_roles_json(&self) -> Result<String, JsValue> {
        self.inner
            .column_roles_json()
            .map_err(|error| JsValue::from_str(&error))
    }

    #[wasm_bindgen(js_name = setRowRange)]
    pub fn set_row_range(&mut self, start: u32, end: u32) -> Result<(), JsValue> {
        self.inner
            .set_row_range(start, end)
            .map_err(|error| JsValue::from_str(&error))
    }

    #[wasm_bindgen(js_name = selectionJson)]
    pub fn selection_json(&self) -> Result<String, JsValue> {
        self.inner
            .selection_json()
            .map_err(|error| JsValue::from_str(&error))
    }

    #[wasm_bindgen(js_name = modelInputJson)]
    pub fn model_input_json(&self, task: &str) -> Result<String, JsValue> {
        self.inner
            .model_input_json(task)
            .map_err(|error| JsValue::from_str(&error))
    }

    #[wasm_bindgen(js_name = runModelJson)]
    pub fn run_model_json(
        &self,
        task: &str,
        algorithm: &str,
        options_json: &str,
    ) -> Result<String, JsValue> {
        let inputs = self
            .inner
            .model_inputs(task)
            .map_err(|error| JsValue::from_str(&error))?;
        models::run_model(inputs, algorithm, options_json)
            .map_err(|error| JsValue::from_str(&error))
    }
}

impl Dataset {
    #[allow(dead_code)]
    pub(crate) fn prepare_model_inputs(&self, task: &str) -> Result<ModelInputs, String> {
        self.inner.model_inputs(task)
    }
}
