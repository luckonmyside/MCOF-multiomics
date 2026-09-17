#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(mixOmics)
  library(pROC)
  library(data.table)
})

# Compatibility shim for older mixOmics code that calls
# dplyr helpers without namespace qualification.
n <- dplyr::n
row_number <- dplyr::row_number
group_by <- dplyr::group_by
filter <- dplyr::filter

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  pos <- which(args == name)
  if (length(pos) == 0) return(default)
  if (pos[length(pos)] == length(args)) {
    stop(paste("Missing value for", name))
  }
  args[pos[length(pos)] + 1]
}

data_root <- get_arg(
  "--data_root",
  "/path/to/mcof-workspace/MCOF_fixed_v1_run/data"
)
split_root <- get_arg(
  "--split_root",
  "/path/to/mcof-workspace/MCOF_baselines_20260722/splits"
)
output_root <- get_arg(
  "--output_root",
  "/path/to/mcof-workspace/MCOF_baselines_20260722/DIABLO"
)

datasets <- c("BRCA", "STAD", "ROSMAP", "SCZ")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

safe_divide <- function(numerator, denominator) {
  ifelse(denominator == 0, 0, numerator / denominator)
}

median_impute <- function(train_matrix, test_matrix) {
  train_matrix <- as.matrix(train_matrix)
  test_matrix <- as.matrix(test_matrix)

  medians <- apply(
    train_matrix,
    2,
    function(x) {
      value <- median(x, na.rm = TRUE)
      if (is.na(value)) 0 else value
    }
  )

  for (j in seq_len(ncol(train_matrix))) {
    train_matrix[is.na(train_matrix[, j]), j] <- medians[j]
    test_matrix[is.na(test_matrix[, j]), j] <- medians[j]
  }

  list(train = train_matrix, test = test_matrix)
}

read_blocks <- function(dataset_dir) {
  blocks <- list()
  index <- 1

  repeat {
    file <- file.path(dataset_dir, paste0(index, "_all.csv"))
    if (!file.exists(file)) break

    matrix_data <- as.matrix(
      fread(file, header = FALSE, data.table = FALSE)
    )

    feature_file <- file.path(
      dataset_dir,
      paste0(index, "_featname.csv")
    )

    if (file.exists(feature_file)) {
      feature_names <- as.character(
        fread(
          feature_file,
          header = FALSE,
          data.table = FALSE
        )[[1]]
      )
    } else {
      feature_names <- paste0(
        "omics",
        index,
        "_f",
        seq_len(ncol(matrix_data)) - 1
      )
    }

    if (length(feature_names) != ncol(matrix_data)) {
      stop(
        paste(
          "Feature-name dimension mismatch:",
          dataset_dir,
          index,
          length(feature_names),
          ncol(matrix_data)
        )
      )
    }

    colnames(matrix_data) <- make.unique(feature_names)
    blocks[[paste0("omics", index)]] <- matrix_data
    index <- index + 1
  }

  if (length(blocks) == 0) {
    stop(paste("No omics files found:", dataset_dir))
  }

  blocks
}

extract_ncomp <- function(perf_object, fallback = 1) {
  value <- tryCatch(
    perf_object$choice.ncomp$WeightedVote[
      "Overall.BER",
      "centroids.dist"
    ],
    error = function(e) NA
  )

  value <- suppressWarnings(as.integer(value))
  if (length(value) == 0 || is.na(value) || value < 1) {
    fallback
  } else {
    value
  }
}

binary_metrics <- function(y_true, y_pred, positive_score, levels_all) {
  y_true <- factor(y_true, levels = levels_all)
  y_pred <- factor(y_pred, levels = levels_all)
  cm <- table(y_true, y_pred)

  tn <- cm[1, 1]
  fp <- cm[1, 2]
  fn <- cm[2, 1]
  tp <- cm[2, 2]

  precision <- safe_divide(tp, tp + fp)
  recall <- safe_divide(tp, tp + fn)
  f1 <- safe_divide(2 * precision * recall, precision + recall)
  acc <- safe_divide(tp + tn, sum(cm))

  denominator <- sqrt(
    (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
  )
  mcc <- ifelse(
    denominator == 0,
    0,
    (tp * tn - fp * fn) / denominator
  )

  roc_object <- pROC::roc(
    response = as.integer(y_true == levels_all[2]),
    predictor = positive_score,
    quiet = TRUE,
    direction = "<"
  )

  data.frame(
    acc = as.numeric(acc),
    f1 = as.numeric(f1),
    auc = as.numeric(pROC::auc(roc_object)),
    mcc = as.numeric(mcc),
    precision = as.numeric(precision),
    recall = as.numeric(recall),
    stringsAsFactors = FALSE
  )
}

multiclass_metrics <- function(y_true, y_pred, score_matrix, levels_all) {
  y_true <- factor(y_true, levels = levels_all)
  y_pred <- factor(y_pred, levels = levels_all)
  cm <- table(y_true, y_pred)

  supports <- rowSums(cm)
  precision <- safe_divide(diag(cm), colSums(cm))
  recall <- safe_divide(diag(cm), rowSums(cm))
  f1_each <- safe_divide(
    2 * precision * recall,
    precision + recall
  )

  auc_each <- rep(NA_real_, length(levels_all))
  names(auc_each) <- levels_all

  for (i in seq_along(levels_all)) {
    binary_true <- as.integer(y_true == levels_all[i])
    if (length(unique(binary_true)) < 2) next

    roc_object <- pROC::roc(
      response = binary_true,
      predictor = score_matrix[, i],
      quiet = TRUE,
      direction = "<"
    )
    auc_each[i] <- as.numeric(pROC::auc(roc_object))
  }

  valid <- !is.na(auc_each)
  macro_auc <- mean(auc_each[valid])
  weighted_auc <- sum(
    auc_each[valid] * supports[valid]
  ) / sum(supports[valid])

  data.frame(
    acc = as.numeric(sum(diag(cm)) / sum(cm)),
    f1_macro = as.numeric(mean(f1_each)),
    f1_weighted = as.numeric(
      sum(f1_each * supports) / sum(supports)
    ),
    auc_macro_ovr = as.numeric(macro_auc),
    auc_weighted_ovr = as.numeric(weighted_auc),
    stringsAsFactors = FALSE
  )
}

all_metrics <- list()
all_markers <- list()

for (dataset in datasets) {
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("Dataset:", dataset, "\n")
  cat(strrep("=", 80), "\n")

  dataset_dir <- file.path(data_root, dataset)
  blocks_all <- read_blocks(dataset_dir)

  raw_labels <- fread(
    file.path(dataset_dir, "labels_all.csv"),
    header = FALSE,
    data.table = FALSE
  )[[1]]

  sorted_levels <- sort(unique(as.character(raw_labels)))
  labels_all <- factor(
    as.character(raw_labels),
    levels = sorted_levels
  )

  splits <- fread(
    file.path(split_root, paste0(dataset, "_outer_splits.csv")),
    data.table = FALSE
  )

  for (repeat_index in 1:5) {
    cat("Repeat", repeat_index, "\n")

    current <- splits[splits[["repeat"]] == repeat_index, ]
    train_idx <- current$sample_index[
      current$subset == "train"
    ] + 1
    test_idx <- current$sample_index[
      current$subset == "test"
    ] + 1

    train_blocks <- list()
    test_blocks <- list()

    for (block_name in names(blocks_all)) {
      imputed <- median_impute(
        blocks_all[[block_name]][train_idx, , drop = FALSE],
        blocks_all[[block_name]][test_idx, , drop = FALSE]
      )
      train_blocks[[block_name]] <- imputed$train
      test_blocks[[block_name]] <- imputed$test
    }

    y_train <- droplevels(labels_all[train_idx])
    y_test <- factor(
      labels_all[test_idx],
      levels = levels(labels_all)
    )

    design <- matrix(
      0.1,
      nrow = length(train_blocks),
      ncol = length(train_blocks),
      dimnames = list(
        names(train_blocks),
        names(train_blocks)
      )
    )
    diag(design) <- 0

    max_ncomp <- min(
      5,
      max(1, nrow(train_blocks[[1]]) - 1),
      min(vapply(train_blocks, ncol, integer(1)))
    )

    basic_model <- block.splsda(
      X = train_blocks,
      Y = y_train,
      ncomp = max_ncomp,
      design = design,
      scale = TRUE
    )

    perf_object <- perf(
      basic_model,
      validation = "Mfold",
      folds = 5,
      nrepeat = 1,
      dist = "centroids.dist",
      progressBar = FALSE
    )

    selected_ncomp <- min(
      max_ncomp,
      extract_ncomp(
        perf_object,
        fallback = min(
          max_ncomp,
          max(1, length(levels(y_train)) - 1)
        )
      )
    )

    keepx_grid <- unique(
      c(5:9, seq(10, 18, 2), seq(20, 30, 5))
    )
    min_features <- min(
      vapply(train_blocks, ncol, integer(1))
    )
    keepx_grid <- keepx_grid[keepx_grid <= min_features]
    if (length(keepx_grid) == 0) {
      keepx_grid <- min_features
    }

    test_keepx <- lapply(
      train_blocks,
      function(x) keepx_grid
    )

    tune_object <- tryCatch(
      tune.block.splsda(
        X = train_blocks,
        Y = y_train,
        ncomp = selected_ncomp,
        test.keepX = test_keepx,
        design = design,
        validation = "Mfold",
        folds = 5,
        nrepeat = 1,
        dist = "centroids.dist",
        progressBar = FALSE
      ),
      error = function(e) {
        message(
          "Tuning failed for ",
          dataset,
          " repeat ",
          repeat_index,
          ": ",
          conditionMessage(e)
        )
        NULL
      }
    )

    if (is.null(tune_object)) {
      fallback_keepx <- lapply(
        train_blocks,
        function(x) {
          rep(
            min(20, ncol(x)),
            selected_ncomp
          )
        }
      )
      selected_keepx <- fallback_keepx
      tuning_status <- "fallback_keepX_20"
    } else {
      selected_keepx <- tune_object$choice.keepX
      tuning_status <- "tuned"
    }

    final_model <- block.splsda(
      X = train_blocks,
      Y = y_train,
      ncomp = selected_ncomp,
      keepX = selected_keepx,
      design = design,
      scale = TRUE
    )

    prediction <- predict(
      final_model,
      newdata = test_blocks
    )

    vote_matrix <- prediction$WeightedVote$centroids.dist
    if (is.null(dim(vote_matrix))) {
      predicted_labels <- vote_matrix
    } else {
      predicted_labels <- vote_matrix[, selected_ncomp]
    }

    score_array <- prediction$WeightedPredict
    if (length(dim(score_array)) == 3) {
      score_matrix <- score_array[, , selected_ncomp]
    } else {
      score_matrix <- score_array
    }

    score_matrix <- as.matrix(score_matrix)
    colnames(score_matrix) <- levels(labels_all)

    predicted_labels <- factor(
      as.character(predicted_labels),
      levels = levels(labels_all)
    )

    if (length(levels(labels_all)) == 2) {
      metrics <- binary_metrics(
        y_true = y_test,
        y_pred = predicted_labels,
        positive_score = score_matrix[, 2],
        levels_all = levels(labels_all)
      )
    } else {
      metrics <- multiclass_metrics(
        y_true = y_test,
        y_pred = predicted_labels,
        score_matrix = score_matrix,
        levels_all = levels(labels_all)
      )
    }

    metrics$dataset <- dataset
    metrics$model <- "DIABLO"
    metrics[["repeat"]] <- repeat_index
    metrics$n_train <- length(train_idx)
    metrics$n_test <- length(test_idx)
    metrics$ncomp <- selected_ncomp
    metrics$tuning_status <- tuning_status

    all_metrics[[length(all_metrics) + 1]] <- metrics

    run_dir <- file.path(
      output_root,
      dataset,
      "DIABLO",
      paste0("repeat_", repeat_index)
    )
    dir.create(
      run_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )

    prediction_table <- data.frame(
      sample_index = test_idx - 1,
      y_true = as.character(y_test),
      y_pred = as.character(predicted_labels),
      stringsAsFactors = FALSE
    )

    for (class_index in seq_along(levels(labels_all))) {
      prediction_table[[
        paste0("score_class_", class_index - 1)
      ]] <- score_matrix[, class_index]
    }

    fwrite(
      prediction_table,
      file.path(run_dir, "predictions_test.csv")
    )

    marker_rows <- list()
    for (block_name in names(train_blocks)) {
      for (component in seq_len(selected_ncomp)) {
        selected <- selectVar(
          final_model,
          block = block_name,
          comp = component
        )[[block_name]]

        if (!is.null(selected$name)) {
          marker_row <- data.frame(
            dataset = dataset,
            block = block_name,
            component = component,
            feature = selected$name,
            stringsAsFactors = FALSE
          )
          marker_row[["repeat"]] <- repeat_index
          marker_row <- marker_row[
            ,
            c(
              "dataset",
              "repeat",
              "block",
              "component",
              "feature"
            ),
            drop = FALSE
          ]
          marker_rows[[length(marker_rows) + 1]] <- marker_row
        }
      }
    }

    if (length(marker_rows)) {
      markers <- rbindlist(marker_rows, fill = TRUE)
      fwrite(
        markers,
        file.path(run_dir, "selected_markers.csv")
      )
      all_markers[[length(all_markers) + 1]] <- markers
    }

    fwrite(
      metrics,
      file.path(run_dir, "metrics_test.csv")
    )
  }
}

metrics_table <- rbindlist(all_metrics, fill = TRUE)
fwrite(
  metrics_table,
  file.path(output_root, "DIABLO_repeat_metrics.csv")
)

metric_columns <- setdiff(
  names(metrics_table),
  c(
    "dataset",
    "model",
    "repeat",
    "tuning_status"
  )
)

summary_rows <- list()

for (dataset in unique(metrics_table$dataset)) {
  subset_data <- metrics_table[
    metrics_table$dataset == dataset,
  ]

  for (metric in metric_columns) {
    values <- suppressWarnings(
      as.numeric(subset_data[[metric]])
    )
    values <- values[!is.na(values)]

    if (length(values) == 0) next

    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      dataset = dataset,
      model = "DIABLO",
      metric = metric,
      n_repeats = length(values),
      mean = mean(values),
      std = ifelse(
        length(values) > 1,
        sd(values),
        0
      ),
      mean_sd_3dp = sprintf(
        "%.3f ± %.3f",
        mean(values),
        ifelse(length(values) > 1, sd(values), 0)
      ),
      stringsAsFactors = FALSE
    )
  }
}

summary_table <- rbindlist(summary_rows, fill = TRUE)
fwrite(
  summary_table,
  file.path(output_root, "DIABLO_summary_long.csv")
)

if (length(all_markers)) {
  marker_table <- rbindlist(all_markers, fill = TRUE)
  fwrite(
    marker_table,
    file.path(output_root, "DIABLO_selected_markers_all.csv")
  )
}

cat("\nAll DIABLO runs completed successfully.\n")
cat("Output:", normalizePath(output_root), "\n")
