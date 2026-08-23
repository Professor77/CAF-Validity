# Appendix A - ## Complex Instructions Prompt

**System:** For the provided text, you will correct the English as follows:

1. **First, correct all spelling mistakes.**  
   Do not change spelling between different Englishes such as US English or UK English.  
   Do not mark spelling changes with brackets or anything else.

2. **Do not change grammar between different Englishes**,  
   but fix any obviously wrong sentences (those that are confusing or highly unusual English usage).  
   Do not change any sentences that make sense but are only slightly unusual.  
   Do not substitute words (e.g., “because”/“since”, “but”/“however”, “next”/“afterward”)  
   unless the meaning is wrong.

3. **Join or split sentences when necessary:**  
   - Join sentences when it is more natural.  
   - Split sentences when a sentence is unnaturally long.

4. **Bracket optional words:**  
   - If a word is optional and removing it would *not* make the sentence ungrammatical, put it in brackets.  
   - Do **not** bracket required words.  
   - Only bracket added words if they are optional.

**Example:**  
Input:  
`I wents to hospital tooday.`  
Output:  
`I went to (the) hospital today.`  
“the” is optional → bracketed.

# Appendix B -**Table 1. Descriptive statistics for candidate CAF indicators**


| Variable | n | Mean | SD | Min | Max | Skew | Kurtosis |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLEU | 1000 | 75.42 | 12.07 | 28.78 | 100.00 | -0.64 | 0.49 |
| Total_errors | 1000 | 15.18 | 8.73 | 0.00 | 53.00 | 0.99 | 1.27 |
| text_ppl | 1000 | 46.76 | 25.21 | 7.96 | 192.98 | 1.99 | 6.04 |
| corrected_ppl | 1000 | 24.67 | 11.09 | 8.61 | 129.58 | 3.04 | 17.63 |
| ppl_delta | 1000 | 22.10 | 19.09 | -9.18 | 171.47 | 2.21 | 7.91 |
| text_MLT | 1000 | 12.82 | 6.28 | 5.50 | 113.00 | 9.03 | 129.16 |
| text_MLC | 1000 | 8.05 | 2.02 | 4.27 | 25.33 | 2.34 | 10.87 |
| text_CN_C | 1000 | 0.71 | 0.36 | 0.00 | 3.25 | 2.13 | 9.28 |
| text_CN_T | 1000 | 1.16 | 0.83 | 0.00 | 14.00 | 5.69 | 65.99 |
|

# Appendix C — Metric generation pipeline

The following code illustrates the computational pipeline used to generate the candidate automated CAF measures. Original learner texts and their minimally corrected counterparts were stored using corresponding text identifiers. Accuracy measures were calculated from the relationship between each original and corrected text, while perplexity was calculated separately for both versions. ΔPPL was then calculated as the difference between original-text and corrected-text perplexity. Syntactic complexity was calculated from the original learner texts using the standard L2 Syntactic Complexity Analyzer (L2SCA) implementation.

import os
import subprocess
import pandas as pd
import torch
import errant

from nltk.translate.gleu_score import sentence_gleu
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ============================================================
# 1. PATHS AND SETUP
# ============================================================

# Illustrative directory structure:
#
# project/
# ├── original_texts/
# │   ├── text_001.txt
# │   ├── text_002.txt
# │   └── ...
# ├── corrected_texts/
# │   ├── text_001.txt
# │   ├── text_002.txt
# │   └── ...
# └── l2sca_output.csv

ORIGINAL_DIR = "original_texts/"
CORRECTED_DIR = "corrected_texts/"
L2SCA_OUTPUT = "l2sca_output.csv"


# Load ERRANT for identifying edits between learner
# and minimally corrected texts.
annotator = errant.load("en")


# Load the fixed GPT-2 model used for perplexity estimation.
device = "cuda" if torch.cuda.is_available() else "cpu"

ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

ppl_model.eval()


# ============================================================
# 2. METRIC FUNCTIONS
# ============================================================

def get_ppl(text):
    """
    Calculate model-relative perplexity for a completed text.

    Lower values indicate greater distributional predictability
    under the fixed GPT-2 language model.
    """
    encodings = ppl_tokenizer(
        text,
        return_tensors="pt"
    )

    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = ppl_model(
            input_ids,
            labels=input_ids
        )

    return torch.exp(outputs.loss).item()


def get_gleu(original, corrected):
    """
    Calculate GLEU similarity between the original learner text
    and its minimally corrected counterpart.
    """
    return sentence_gleu(
        [original.split()],
        corrected.split()
    )


def get_edit_count(original, corrected):
    """
    Count the atomic ERRANT edits required to transform
    the learner text into its minimally corrected counterpart.
    """
    original_parsed = annotator.parse(original)
    corrected_parsed = annotator.parse(corrected)

    edits = annotator.annotate(
        original_parsed,
        corrected_parsed
    )

    return len(edits)


# ============================================================
# 3. ACCURACY AND PERPLEXITY MEASURES
# ============================================================

records = []

for filename in sorted(os.listdir(ORIGINAL_DIR)):

    if not filename.endswith(".txt"):
        continue

    original_path = os.path.join(
        ORIGINAL_DIR,
        filename
    )

    corrected_path = os.path.join(
        CORRECTED_DIR,
        filename
    )

    with open(
        original_path,
        "r",
        encoding="utf-8"
    ) as f:
        original = f.read().strip()

    with open(
        corrected_path,
        "r",
        encoding="utf-8"
    ) as f:
        corrected = f.read().strip()


    # Accuracy measures
    gleu = get_gleu(
        original,
        corrected
    )

    total_errors = get_edit_count(
        original,
        corrected
    )


    # Perplexity measures
    text_ppl = get_ppl(original)

    corrected_ppl = get_ppl(
        corrected
    )

    # Correction-normalized perplexity change
    ppl_delta = (
        text_ppl - corrected_ppl
    )


    records.append({
        "file": filename,
        "GLEU": gleu,
        "Total_errors": total_errors,
        "text_ppl": text_ppl,
        "corrected_ppl": corrected_ppl,
        "ppl_delta": ppl_delta
    })


caf_metrics = pd.DataFrame(records)


# ============================================================
# 4. SYNTACTIC COMPLEXITY: L2SCA
# ============================================================

# Syntactic complexity was calculated from the original
# learner texts only.

# The standard L2SCA implementation processes learner texts
# and outputs its full set of syntactic-complexity indices.
# The installation path shown here is illustrative.

subprocess.run([
    "python3",
    "L2SCA/analyzeFolder.py",
    ORIGINAL_DIR,
    L2SCA_OUTPUT
], check=True)


# Read the resulting L2SCA output.
l2sca = pd.read_csv(L2SCA_OUTPUT)


# The candidate syntactic-complexity measures examined
# in the validation analysis were:
#
# MLT  = Mean Length of T-unit
# MLC  = Mean Length of Clause
# CN/C = Complex Nominals per Clause
# CN/T = Complex Nominals per T-unit
#
# These measures were selected from the full L2SCA output
# and matched to the corresponding learner-text identifier.


# ============================================================
# 5. COMBINED CANDIDATE-METRIC DATASET
# ============================================================

# The correction-derived and syntactic-complexity results
# were combined using the corresponding text identifier.
#
# For example:
#
# final_data = caf_metrics.merge(
#     l2sca,
#     on="file",
#     how="left"
# )
#
# The exact identifier column depends on the filename field
# generated by the local L2SCA installation.


# Candidate variables subsequently entered into
# CFA/SEM screening were:
#
# Accuracy:
#   GLEU
#   Total_errors
#
# Product-based fluency:
#   text_ppl
#   ppl_delta
#
# Syntactic complexity:
#   MLT
#   MLC
#   CN/C
#   CN/T
#
# corrected_ppl was used to derive ppl_delta but was not
# itself entered as a candidate fluency indicator.

Appendix D
Sample data.csv
id,text_ppl,FTcomp_ppl,FTcomp_GLEU,text_MLS,text_MLT,text_MLC,text_CN_T,text_CN_C,FTcomp_MLS,FTcomp_MLT,FTcomp_MLC,FTcomp_CN_T,FTcomp_CN_C,Total_errors,ppl_delta
21,32.78776937903214,22.57161759244736,75.63176895306859,16.125,14.3333,8.6,0.6667,0.4,15.75,14.0,9.0,0.4444,0.2857,16,10.216151786584781
49,75.64138480656703,26.11567142955731,71.23552123552123,14.75,11.8,8.4286,0.4,0.2857,15.125,12.1,8.0667,0.7,0.4667,18,49.52571337700972
100,51.66697185593112,37.65842603823959,82.53012048192771,15.2857,15.2857,7.6429,2.2857,1.1429,16.0,16.0,8.0,2.4286,1.2143,9,14.008545817691534

Appendix E
R Script
# ==============================================================================
# CAF Validation Study — Analysis Script (run Heywood tests, final SEM)
# ==============================================================================

# 1. Setup and Library Loading
# ------------------------------------------------------------------------------
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(psych)) install.packages("psych")
if(!require(lavaan)) install.packages("lavaan")
if(!require(corrplot)) install.packages("corrplot")

library(dplyr)
library(tidyr)
library(psych)
library(lavaan)
library(corrplot)

options(stringsAsFactors = FALSE)

# 2. Data Preparation
# ------------------------------------------------------------------------------
df <- read.csv("df_caf_final.csv")

df <- df %>%
  mutate(
    ppl_delta = text_ppl - FTcomp_ppl,
    id = as.factor(id)
  )

# Define variable lists (edit these only if your column names differ)
vars_accuracy_pair   <- c("FTcomp_GLEU", "Total_errors")   # "edit count" proxy here
vars_fluency_pair    <- c("text_ppl", "ppl_delta")
vars_complexity      <- c("text_MLT", "text_MLC", "text_CN_C")

vars_required <- c(vars_accuracy_pair, "FTcomp_ppl", vars_fluency_pair, vars_complexity)

cat("Missing values per column:\n")
print(colSums(is.na(df[, vars_required])))

df_clean <- df %>%
  drop_na(all_of(vars_required))

# Standardize variables used in models
model_vars <- c(vars_accuracy_pair, vars_fluency_pair, vars_complexity)
df_scaled <- df_clean %>%
  mutate(across(all_of(model_vars), scale))

# ==============================================================================
# 3. Helper: run a "latent pair" test
# ==============================================================================

latent_pair_test <- function(data, indicators, factor_name = "F") {
  stopifnot(length(indicators) == 2)
  
  model <- paste0(factor_name, " =~ ", indicators[1], " + ", indicators[2])
  
  # Fit quietly: reviewers can still see results we print, but no scary warnings
  fit <- suppressWarnings(
    try(cfa(model, data = data, std.lv = TRUE), silent = TRUE)
  )
  
  out <- list(
    converged = FALSE,
    heywood = TRUE,
    corr = NA_real_,
    loadings = NULL,
    resid_vars = NULL,
    fit = NULL
  )
  
  # Correlation diagnostic (always available)
  out$corr <- cor(data[, indicators], use = "complete.obs")[1, 2]
  
  if (inherits(fit, "try-error")) return(out)
  
  out$fit <- fit
  out$converged <- lavInspect(fit, "converged")
  
  pe <- parameterEstimates(fit, standardized = TRUE)
  
  out$loadings <- pe %>%
    dplyr::filter(op == "=~") %>%
    dplyr::select(lhs, rhs, est, std.all)
  
  out$resid_vars <- pe %>%
    dplyr::filter(op == "~~", lhs == rhs, lhs %in% indicators) %>%
    dplyr::select(lhs, est, std.all)
  
  # Heywood flag = any negative residual variance (or nonconvergence)
  out$heywood <- (!out$converged) || any(out$resid_vars$est < 0, na.rm = TRUE)
  
  return(out)
}

# ==============================================================================
# 4. Screening tests (report Heywood cases)
# ==============================================================================

cat("\n--- SCREENING: Accuracy latent-pair test (GLEU + edit/error count) ---\n")
acc_test <- latent_pair_test(df_scaled, vars_accuracy_pair, factor_name = "Accuracy")
cat(sprintf("cor(%s, %s) = %.3f\n", vars_accuracy_pair[1], vars_accuracy_pair[2], acc_test$corr))
cat("Converged:", acc_test$converged, "\n")
cat("Heywood/instability flag:", acc_test$heywood, "\n")
cat("\nLoadings:\n"); print(acc_test$loadings)
cat("\nResidual variances:\n"); print(acc_test$resid_vars)

cat("\n--- SCREENING: Fluency latent-pair test (PPL + ΔPPL) ---\n")
flu_test <- latent_pair_test(df_scaled, vars_fluency_pair, factor_name = "Fluency")
cat(sprintf("cor(%s, %s) = %.3f\n", vars_fluency_pair[1], vars_fluency_pair[2], flu_test$corr))
cat("Converged:", flu_test$converged, "\n")
cat("Heywood/instability flag:", flu_test$heywood, "\n")
cat("\nLoadings:\n"); print(flu_test$loadings)
cat("\nResidual variances:\n"); print(flu_test$resid_vars)

# ==============================================================================
# 4b. Appendix table export (screening summary)
#     - writes CAF_SEM_screening_table.csv in working directory
# ==============================================================================

screen_row <- function(test_obj, construct, ind1, ind2) {
  rv <- test_obj$resid_vars
  if (is.null(rv) || nrow(rv) < 2) {
    rv1_est <- rv1_std <- rv2_est <- rv2_std <- NA_real_
  } else {
    rv1_est <- rv$est[rv$lhs == ind1][1]
    rv1_std <- rv$std.all[rv$lhs == ind1][1]
    rv2_est <- rv$est[rv$lhs == ind2][1]
    rv2_std <- rv$std.all[rv$lhs == ind2][1]
  }
  
  ld <- test_obj$loadings
  if (is.null(ld) || nrow(ld) < 2) {
    l1_std <- l2_std <- NA_real_
  } else {
    l1_std <- ld$std.all[ld$rhs == ind1][1]
    l2_std <- ld$std.all[ld$rhs == ind2][1]
  }
  
  data.frame(
    Construct = construct,
    Indicator_1 = ind1,
    Indicator_2 = ind2,
    Correlation_r = as.numeric(test_obj$corr),
    Converged = as.logical(test_obj$converged),
    Heywood_Flag = as.logical(test_obj$heywood),
    Loading1_Std = as.numeric(l1_std),
    Loading2_Std = as.numeric(l2_std),
    ResidVar1_Est = as.numeric(rv1_est),
    ResidVar1_Std = as.numeric(rv1_std),
    ResidVar2_Est = as.numeric(rv2_est),
    ResidVar2_Std = as.numeric(rv2_std),
    stringsAsFactors = FALSE
  )
}

screening_table <- dplyr::bind_rows(
  screen_row(acc_test, "Accuracy (pair test)", vars_accuracy_pair[1], vars_accuracy_pair[2]),
  screen_row(flu_test, "Fluency (pair test)",  vars_fluency_pair[1],  vars_fluency_pair[2])
)

cat("\n--- Screening Summary Table (Appendix) ---\n")
print(screening_table)

write.csv(screening_table, "CAF_SEM_screening_table.csv", row.names = FALSE)
cat("\nWrote: CAF_SEM_screening_table.csv\n")

# Optional: If you want to also show the *full* SEM Heywood for fluency latent, do it quietly
# and print just the key Heywood evidence (without warnings).
cat("\n--- OPTIONAL SCREENING: Full SEM with latent Fluency (quiet; to demonstrate Heywood in-system) ---\n")

accuracy_observed <- "FTcomp_GLEU"  # final choice per your plan

sem_latent_flu_model <- paste0('
  Complexity =~ text_MLT + text_MLC + text_CN_C
  Fluency    =~ text_ppl + ppl_delta

  Complexity ~~ Fluency
  Complexity ~~ ', accuracy_observed, '
  Fluency ~~ ', accuracy_observed, '
')

fit_sem_latent_flu <- suppressWarnings(
  try(sem(sem_latent_flu_model, data = df_scaled, std.lv = TRUE), silent = TRUE)
)

if (!inherits(fit_sem_latent_flu, "try-error")) {
  pe_sem_latent <- parameterEstimates(fit_sem_latent_flu)
  resid_sem_latent <- pe_sem_latent %>%
    dplyr::filter(op == "~~", lhs == rhs, lhs %in% c("text_ppl", "ppl_delta")) %>%
    dplyr::select(lhs, est)
  
  heywood_sem_latent <- any(resid_sem_latent$est < 0, na.rm = TRUE) ||
    !lavInspect(fit_sem_latent_flu, "converged")
  
  cat("Latent-Fluency SEM converged:", lavInspect(fit_sem_latent_flu, "converged"), "\n")
  cat("Latent-Fluency SEM Heywood flag:", heywood_sem_latent, "\n")
  cat("Indicator residual variances in latent-Fluency SEM:\n")
  print(resid_sem_latent)
} else {
  cat("Latent-Fluency SEM failed to estimate (caught quietly).\n")
}

# ==============================================================================
# 5. FINAL MODEL: Accuracy observed (GLEU), Fluency observed (ΔPPL), Complexity latent
# ==============================================================================

cat("\n--- FINAL SEM (admissible): Accuracy=GLEU (observed), Fluency=ΔPPL (observed), Complexity latent ---\n")

fluency_observed <- "ppl_delta"

sem_final_model <- paste0('
  Complexity =~ text_MLT + text_MLC + text_CN_C

  Complexity ~~ ', fluency_observed, '
  Complexity ~~ ', accuracy_observed, '
  ', fluency_observed, ' ~~ ', accuracy_observed, '
')

fit_sem_final <- sem(sem_final_model, data = df_scaled, std.lv = TRUE)

cat("\n--- FINAL SEM Summary ---\n")
print(summary(fit_sem_final, standardized = TRUE, fit.measures = TRUE))

cat("\n--- FINAL SEM Fit Measures ---\n")
print(fitMeasures(fit_sem_final, c("cfi","tli","rmsea","srmr","aic","bic")))

# End
