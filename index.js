import init, { Dataset } from "./pkg/foundml_wasm.js?v=4";

const $ = (selector) => document.querySelector(selector);
const fileInput = $("#file-input");
const fileDrop = $("#file-drop");
const datasetTable = $("#dataset-table");
const datasetStatus = $("#dataset-status");
const wasmStatus = $("#wasm-status");
const dataSummary = $("#data-summary");
const rowStart = $("#row-start");
const rowEnd = $("#row-end");
const rangeStatus = $("#range-status");
const prepareStatus = $("#prepare-status");
const preparedOutput = $("#prepared-output");
const resultMetrics = $("#result-metrics");
const resultExtra = $("#result-extra");
const resultOutput = $("#result-output");
const resultCanvas = $("#results-chart");
const task = $("#task");
const algorithm = $("#algorithm");

const algorithms = {
  regression: [
    ["linear_regression", "Linear regression"],
    ["decision_tree_regressor", "Decision tree"],
    ["random_forest_regressor", "Random forest"],
    ["knn_regressor", "KNN"],
    ["poisson_glm", "Poisson GLM"],
  ],
  classification: [
    ["logistic_regression", "Logistic regression"],
    ["decision_tree_classifier", "Decision tree"],
    ["random_forest_classifier", "Random forest"],
    ["knn_classifier", "KNN"],
    ["perceptron", "Perceptron"],
  ],
  clustering: [
    ["kmeans", "K-means"],
    ["hierarchical", "Hierarchical (single linkage)"],
  ],
  dimensionality_reduction: [
    ["pca", "PCA"],
    ["svd", "SVD"],
  ],
};

const colors = ["#1a5fb4", "#c64600", "#26a269", "#a51d2d", "#813d9c"];
let dataset = null;
let snapshot = null;
let currentFile = null;
let wasmReady = false;
let lastPlot = null;

function setStatus(element, message, kind = "info") {
  element.textContent = message;
  element.dataset.kind = kind;
}

function setDatasetControlsEnabled(enabled) {
  document.querySelectorAll("[data-needs-dataset]").forEach((control) => {
    control.disabled = !enabled;
  });
}

function clearPlot() {
  const context = resultCanvas.getContext("2d");
  const bounds = resultCanvas.getBoundingClientRect();
  const width = Math.max(1, bounds.width);
  const height = Math.max(1, bounds.height);
  const ratio = window.devicePixelRatio || 1;
  resultCanvas.width = Math.round(width * ratio);
  resultCanvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#6b6b6b";
  context.font = "14px system-ui, sans-serif";
  context.fillText("A plot will appear after a model run.", 16, 26);
  lastPlot = null;
}

function clearResults() {
  resultMetrics.replaceChildren();
  resultExtra.replaceChildren();
  resultOutput.textContent = "No model has been run.";
  clearPlot();
}

function clearPreparedData() {
  preparedOutput.textContent = "No model inputs prepared.";
  prepareStatus.textContent = "Load a dataset and select column roles first.";
  clearResults();
}

function clearDataset(message = "No dataset loaded.") {
  dataset = null;
  snapshot = null;
  datasetTable.replaceChildren();
  dataSummary.replaceChildren();
  rowStart.value = "1";
  rowEnd.value = "1";
  rowStart.max = "1";
  rowEnd.max = "1";
  rangeStatus.textContent = "Load a dataset to select rows.";
  setDatasetControlsEnabled(false);
  clearPreparedData();
  setStatus(datasetStatus, message);
}

function updateAlgorithmOptions() {
  algorithm.replaceChildren();
  for (const [value, label] of algorithms[task.value] ?? []) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    algorithm.append(option);
  }
}

function syncColumnRoles() {
  if (!dataset) return;
  const roles = JSON.parse(dataset.columnRolesJson());
  document.querySelectorAll("[data-column-role]").forEach((select) => {
    select.value = roles[Number(select.dataset.columnRole)];
  });
  clearPreparedData();
}

function renderDataset() {
  if (!snapshot) return;

  datasetTable.replaceChildren();
  const head = document.createElement("thead");
  const headerRow = head.insertRow();
  const numberHeader = document.createElement("th");
  numberHeader.textContent = "#";
  headerRow.append(numberHeader);

  for (const column of snapshot.columns) {
    const th = document.createElement("th");
    const title = document.createElement("strong");
    title.textContent = column.name;

    const info = document.createElement("small");
    info.textContent =
      `${column.type} · ${column.distinct_count} unique · ${column.missing_count} blank`;

    const role = document.createElement("select");
    role.dataset.columnRole = String(column.index);
    role.setAttribute("aria-label", `Role for ${column.name}`);
    for (const value of ["feature", "target", "ignore"]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      option.selected = value === column.role;
      role.append(option);
    }
    role.addEventListener("change", () => {
      try {
        dataset.setColumnRole(column.index, role.value);
        syncColumnRoles();
        setStatus(datasetStatus, "Column roles updated.", "success");
      } catch (error) {
        syncColumnRoles();
        setStatus(datasetStatus, String(error), "error");
      }
    });

    th.append(title, info, role);
    headerRow.append(th);
  }

  const body = document.createElement("tbody");
  snapshot.rows.forEach((row, rowIndex) => {
    const tr = body.insertRow();
    tr.insertCell().textContent = String(rowIndex + 1);
    row.forEach((value) => {
      tr.insertCell().textContent = value;
    });
  });
  datasetTable.append(head, body);

  const rowCount = snapshot.row_count;
  rowStart.max = String(Math.max(1, rowCount));
  rowEnd.max = String(Math.max(1, rowCount));
  rowStart.value = "1";
  rowEnd.value = String(Math.max(1, rowCount));
  rangeStatus.textContent =
    rowCount === 0 ? "This file has no data rows." : `All ${rowCount} rows selected.`;
  dataSummary.textContent =
    `${rowCount.toLocaleString()} rows · ${snapshot.column_count} columns · delimiter: ${snapshot.delimiter}`;

  setDatasetControlsEnabled(rowCount > 0);
  clearPreparedData();
}

function updateRowRange() {
  if (!dataset || !snapshot || snapshot.row_count === 0) return;
  try {
    dataset.setRowRange(Number(rowStart.value), Number(rowEnd.value));
    const selection = JSON.parse(dataset.selectionJson());
    rangeStatus.textContent =
      `${selection.selected_row_count} rows selected (rows ${selection.row_start}–${selection.row_end}).`;
    clearPreparedData();
    setStatus(datasetStatus, "Row selection updated.", "success");
  } catch (error) {
    rangeStatus.textContent = String(error);
    setStatus(datasetStatus, String(error), "error");
  }
}

async function loadFile(file) {
  if (!wasmReady) {
    setStatus(wasmStatus, "Rust/WASM is not ready yet.", "error");
    return;
  }
  currentFile = file;
  clearDataset("Reading file…");
  try {
    dataset = new Dataset(
      await file.text(),
      $("#delimiter").value,
      $("#has-headers").checked,
    );
    snapshot = JSON.parse(dataset.snapshotJson());
    renderDataset();
    setStatus(datasetStatus, `${file.name} loaded.`, "success");
  } catch (error) {
    clearDataset(`${file.name}: ${String(error)}`);
    setStatus(datasetStatus, `${file.name}: ${String(error)}`, "error");
  }
}

function appendResultTable(container, headings, rows) {
  const table = document.createElement("table");
  table.className = "confusion-table";
  const header = table.createTHead().insertRow();
  headings.forEach((heading) => {
    const cell = document.createElement("th");
    cell.textContent = String(heading);
    header.append(cell);
  });
  const body = table.createTBody();
  rows.forEach((row) => {
    const tr = body.insertRow();
    row.forEach((value) => {
      tr.insertCell().textContent = String(value);
    });
  });
  container.append(table);
}

function drawEmptyChart() {
  const context = resultCanvas.getContext("2d");
  const bounds = resultCanvas.getBoundingClientRect();
  const width = Math.max(1, bounds.width);
  const height = Math.max(1, bounds.height);
  const ratio = window.devicePixelRatio || 1;
  resultCanvas.width = Math.round(width * ratio);
  resultCanvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#6b6b6b";
  context.font = "14px system-ui, sans-serif";
  context.fillText("A plot will appear after a model run.", 16, 26);
  lastPlot = null;
}

function drawScatterPlot(plot) {
  lastPlot = plot;
  const bounds = resultCanvas.getBoundingClientRect();
  const width = Math.max(1, bounds.width);
  const height = Math.max(1, bounds.height);
  const ratio = window.devicePixelRatio || 1;
  resultCanvas.width = Math.round(width * ratio);
  resultCanvas.height = Math.round(height * ratio);
  const context = resultCanvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const points = plot.points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!points.length) return drawEmptyChart();

  let minX = Math.min(...points.map(([x]) => x));
  let maxX = Math.max(...points.map(([x]) => x));
  let minY = Math.min(...points.map(([, y]) => y));
  let maxY = Math.max(...points.map(([, y]) => y));
  if (plot.identityLine) {
    minX = minY = Math.min(minX, minY);
    maxX = maxY = Math.max(maxX, maxY);
  }
  if (minX === maxX) [minX, maxX] = [minX - 1, maxX + 1];
  if (minY === maxY) [minY, maxY] = [minY - 1, maxY + 1];

  const margin = { left: 54, right: 14, top: 16, bottom: 40 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xPixel = (value) => margin.left + ((value - minX) / (maxX - minX)) * plotWidth;
  const yPixel = (value) => margin.top + ((maxY - value) / (maxY - minY)) * plotHeight;

  context.font = "11px system-ui, sans-serif";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const x = margin.left + (plotWidth * tick) / 4;
    const y = margin.top + (plotHeight * tick) / 4;
    context.strokeStyle = "#e8e8e8";
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, margin.top + plotHeight);
    context.moveTo(margin.left, y);
    context.lineTo(margin.left + plotWidth, y);
    context.stroke();
  }
  context.strokeStyle = "#777";
  context.beginPath();
  context.moveTo(margin.left, margin.top);
  context.lineTo(margin.left, margin.top + plotHeight);
  context.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  context.stroke();

  if (plot.identityLine) {
    context.strokeStyle = "#777";
    context.setLineDash([5, 4]);
    context.beginPath();
    context.moveTo(xPixel(minX), yPixel(minY));
    context.lineTo(xPixel(maxX), yPixel(maxY));
    context.stroke();
    context.setLineDash([]);
  }

  points.forEach(([x, y], index) => {
    const group = plot.groups?.[index] ?? 0;
    context.fillStyle = colors[group % colors.length] + "cc";
    context.beginPath();
    context.arc(xPixel(x), yPixel(y), 3.5, 0, Math.PI * 2);
    context.fill();
  });
  context.fillStyle = "#1c1c1c";
  context.font = "12px system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(plot.xLabel, margin.left + plotWidth / 2, height - 5);
  context.save();
  context.translate(13, margin.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(plot.yLabel, 0, 0);
  context.restore();
}

function renderResults(result) {
  resultMetrics.replaceChildren();
  resultExtra.replaceChildren();
  for (const [name, value] of Object.entries(result.metrics ?? {})) {
    const card = document.createElement("div");
    card.className = "metric-card";
    const strong = document.createElement("strong");
    strong.textContent = typeof value === "number" ? value.toFixed(4) : String(value ?? "—");
    const label = document.createElement("span");
    label.textContent = name.replaceAll("_", " ");
    card.append(strong, label);
    resultMetrics.append(card);
  }

  if (Array.isArray(result.confusion_matrix)) {
    const labels = result.class_labels ?? [];
    const rows = result.confusion_matrix.map((counts, index) => [labels[index] ?? `Class ${index}`, ...counts]);
    appendResultTable(resultExtra, ["Actual / predicted", ...labels], rows);
    drawScatterPlot({
      points: result.actual_class_ids.map((actual, index) => [actual, result.predicted_class_ids[index]]),
      groups: result.actual_class_ids,
      xLabel: "Actual class ID",
      yLabel: "Predicted class ID",
    });
  } else if (Array.isArray(result.predicted)) {
    const values = result.task === "classification"
      ? result.actual_labels.map((actual, index) => [result.test_row_numbers[index], actual, result.predicted_labels[index]])
      : result.actual.map((actual, index) => [result.test_row_numbers[index], actual, result.predicted[index]]);
    appendResultTable(resultExtra, ["Row", "Actual", "Predicted"], values);
    if (result.task === "regression") {
      drawScatterPlot({
        points: result.actual.map((actual, index) => [actual, result.predicted[index]]),
        xLabel: "Actual",
        yLabel: "Predicted",
        identityLine: true,
      });
    }
  } else if (Array.isArray(result.cluster_labels)) {
    for (const [cluster, size] of (result.cluster_sizes ?? []).entries()) {
      const card = document.createElement("div");
      card.className = "metric-card";
      const strong = document.createElement("strong");
      strong.textContent = String(size);
      const label = document.createElement("span");
      label.textContent = `cluster ${cluster} rows`;
      card.append(strong, label);
      resultMetrics.append(card);
    }
    appendResultTable(
      resultExtra,
      ["Row", "Cluster"],
      result.cluster_labels.map((cluster, index) => [result.row_numbers[index], cluster]),
    );
    drawScatterPlot({
      points: result.coordinates,
      groups: result.cluster_labels,
      xLabel: "Feature 1",
      yLabel: "Feature 2",
    });
  } else if (Array.isArray(result.embedding)) {
    appendResultTable(
      resultExtra,
      ["Row", ...(result.component_names ?? [])],
      result.embedding.map((coordinates, index) => [result.row_numbers[index], ...coordinates]),
    );
    drawScatterPlot({
      points: result.embedding.map((coordinates) => [coordinates[0] ?? 0, coordinates[1] ?? 0]),
      xLabel: result.component_names?.[0] ?? "Component 1",
      yLabel: result.component_names?.[1] ?? "Component 2",
    });
  }

  resultOutput.textContent = JSON.stringify(result, null, 2);
}

function modelOptions() {
  return {
    test_percent: Number($("#test-size").value),
    seed: Number($("#seed").value),
    n_trees: Number($("#n-trees").value),
    max_depth: Number($("#max-depth").value),
    k: Number($("#model-k").value),
    n_components: Number($("#n-components").value),
    max_iter: Number($("#max-iter").value),
    epochs: Number($("#epochs").value),
    learning_rate: Number($("#learning-rate").value),
  };
}

function bindEvents() {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) loadFile(file);
  });
  for (const control of [$("#delimiter"), $("#has-headers")]) {
    control.addEventListener("change", () => {
      if (currentFile) loadFile(currentFile);
    });
  }
  fileDrop.addEventListener("dragover", (event) => {
    event.preventDefault();
    fileDrop.classList.add("is-dragging");
  });
  fileDrop.addEventListener("dragleave", () => fileDrop.classList.remove("is-dragging"));
  fileDrop.addEventListener("drop", (event) => {
    event.preventDefault();
    fileDrop.classList.remove("is-dragging");
    const file = event.dataTransfer?.files?.[0];
    if (file) loadFile(file);
  });
  $("#numeric-to-feature").addEventListener("click", () => {
    if (!dataset || !snapshot) return;
    snapshot.columns.forEach((column) => {
      if (column.type === "numeric" || column.type === "boolean") dataset.setColumnRole(column.index, "feature");
    });
    syncColumnRoles();
    setStatus(datasetStatus, "Numeric and boolean columns set as features.", "success");
  });
  $("#all-to-feature").addEventListener("click", () => {
    if (!dataset || !snapshot) return;
    snapshot.columns.forEach((column) => dataset.setColumnRole(column.index, "feature"));
    syncColumnRoles();
    setStatus(datasetStatus, "All columns set as features.", "success");
  });
  $("#all-to-ignore").addEventListener("click", () => {
    if (!dataset || !snapshot) return;
    snapshot.columns.forEach((column) => dataset.setColumnRole(column.index, "ignore"));
    syncColumnRoles();
    setStatus(datasetStatus, "All columns set to ignore.", "success");
  });
  rowStart.addEventListener("change", updateRowRange);
  rowEnd.addEventListener("change", updateRowRange);
  $("#test-size").addEventListener("input", (event) => {
    $("#test-size-label").textContent = `${event.target.value}%`;
    clearPreparedData();
  });
  for (const selector of ["#seed", "#n-trees", "#max-depth", "#model-k", "#n-components", "#max-iter", "#epochs", "#learning-rate"]) {
    $(selector).addEventListener("change", clearPreparedData);
  }
  $("#prepare-data").addEventListener("click", () => {
    if (!dataset) return;
    try {
      const input = JSON.parse(dataset.modelInputJson(task.value));
      preparedOutput.textContent = JSON.stringify({
        task: input.task,
        x_shape: [input.row_count, input.feature_count],
        y_kind: input.y_kind,
        class_labels: input.class_labels,
        feature_names: input.feature_names,
        first_row_numbers: input.row_numbers.slice(0, 5),
        first_x_rows: input.x.slice(0, 5),
        first_y_values: Array.isArray(input.y) ? input.y.slice(0, 5) : null,
      }, null, 2);
      prepareStatus.textContent = `Prepared ${input.row_count} rows × ${input.feature_count} features.`;
      setStatus(datasetStatus, "Model-ready inputs prepared.", "success");
    } catch (error) {
      preparedOutput.textContent = "Unable to prepare model inputs.";
      prepareStatus.textContent = String(error);
      setStatus(datasetStatus, String(error), "error");
    }
  });
  $("#run-model").addEventListener("click", () => {
    if (!dataset) return;
    prepareStatus.textContent = "Running model in Rust/WASM…";
    try {
      if (typeof dataset.runModelJson !== "function") {
        throw new Error("The WASM package is outdated; rebuild with wasm-pack build --target web --dev.");
      }
      const result = JSON.parse(dataset.runModelJson(task.value, algorithm.value, JSON.stringify(modelOptions())));
      renderResults(result);
      prepareStatus.textContent = `${result.algorithm} finished for ${result.task}.`;
      setStatus(datasetStatus, "Model run complete.", "success");
    } catch (error) {
      resultMetrics.replaceChildren();
      resultExtra.replaceChildren();
      resultOutput.textContent = "Model run failed.";
      prepareStatus.textContent = String(error);
      setStatus(datasetStatus, String(error), "error");
    }
  });
  task.addEventListener("change", () => {
    updateAlgorithmOptions();
    clearPreparedData();
  });
  algorithm.addEventListener("change", clearPreparedData);
  window.addEventListener("resize", () => {
    if (lastPlot) drawScatterPlot(lastPlot);
  });
}

async function start() {
  updateAlgorithmOptions();
  setDatasetControlsEnabled(false);
  clearDataset();
  bindEvents();

  try {
    await init({
      module_or_path: new URL("./pkg/foundml_wasm_bg.wasm?v=4", import.meta.url),
    });
    if (typeof Dataset.prototype.runModelJson !== "function") {
      throw new Error("The generated WASM package is outdated. Rebuild with wasm-pack.");
    }
    wasmReady = true;
    setStatus(wasmStatus, "Rust/WASM loaded.", "success");
  } catch (error) {
    setStatus(wasmStatus, `Could not load Rust/WASM: ${String(error)}`, "error");
  }
}

start();
