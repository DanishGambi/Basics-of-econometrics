author: "MALIUZHANTCEV"
date: "`r Sys.Date()`"
output: 
  html_document:
    toc: true
    number_sections: true
    code_folding: hide
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
library(mfx); library(vcd); library(reshape2); library(skimr); library(AUC)
library(rio); library(tidyverse); library(dplyr); library(ggplot2); library(glmnet); library(haven); library(pROC) ; library(margins); library(corrplot);library(lmtest)
```

```{r python-setup, include=FALSE}


library(reticulate)

# Use whatever Python was automatically detected
print(py_config())

# Install matplotlib in this environment
py_install("matplotlib")
```


# Analysis of Marriage Probability Factors

```{r libraries}
library(mfx)  # расчет предельных эффектов
library(vcd)  # графики для качественных данных
library(reshape2)  # манипуляции с данными
library(skimr) # описательные статистики
library(AUC)  # для ROC кривой
library(rio) # импорт файлов разных форматов
library(tidyverse) # графики и манипуляции с данными
library(dplyr) # манипуляции с таблицами
library(ggplot2) 
library(glmnet)
library(haven)
```

## Data Preparation

```{r load_data}
data <- read_dta("Dougherty.dta")

# view(data)
summary(data)
complete_data <- data[complete.cases(data), ]

#HEIGHT, CATGOV - must be used as predictors (Professors' HW requirement)
# "HEIGHT", "CATGOV", "AGE", "EARNINGS", "SIBLINGS", "MARRIED", "TENURE", "EDUCDO", "EDUCMAST" - extra predictors. Chosen based on economic intuition. 
```

## Correlation Analysis

### Heatmap of All Factors

```{r heatmap_all}
df <- complete_data %>%
  mutate(across(c(CATGOV, CATPRI, CATSE, MARRIED), as.numeric))
cor_matrix <- cor(df, use = "complete.obs")
heatmap(cor_matrix, 
        col = colorRampPalette(c("blue", "white", "red"))(100),
        symm = TRUE)
```

### Heatmap of Selected Factors

```{r heatmap_selected}
selected_cols <- c("HEIGHT", "CATGOV", "AGE", "EARNINGS", "SIBLINGS", "MARRIED", "TENURE", "EDUCDO", "EDUCMAST")
cor_data <- complete_data[, selected_cols]
cor_data$MARRIED <- as.numeric(cor_data$MARRIED)
cor_matrix <- cor(cor_data, use = "complete.obs")  # Handles missing values

library(corrplot)

corrplot(cor_matrix,
         method = "color", type = "upper",  tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.7,  
         col = colorRampPalette(c("blue", "white", "red"))(100))

# stick with "AGE", "EARNINGS" and "SIBLINGS" as optional predictors
```

## Model Building

### Data Preparation for Modeling

```{r model_data_prep}
model_data <- complete_data %>%
  mutate(
    MARRIED = as.numeric(MARRIED),   
    CATGOV = as.factor(CATGOV)     
  ) %>%
  select(MARRIED, AGE, EARNINGS, SIBLINGS, HEIGHT, CATGOV) %>%
  na.omit()  
```

### Full Logistic Regression Model

```{r full_model}
model <- glm(MARRIED ~ AGE + EARNINGS + SIBLINGS + HEIGHT + CATGOV,
             data = model_data,
             family = binomial(link = "logit"))  

summary(model)
```

### Constrained Model (HEIGHT + CATGOV only)

```{r constrained_model}
model_constrained <- glm(MARRIED ~ HEIGHT + CATGOV,
             data = model_data,
             family = binomial(link = "logit")) 

summary(model_constrained)
```

### Marginal Effects

```{r marginal_effects}
library(margins)

marg_effects <- margins(model_constrained)
summary(marg_effects)

# chances
exp(coef(model_constrained))
```

## Likelihood Ratio Tests

```{r lrt_tests}
# Full model (with all predictors)
full_model <- glm(MARRIED ~ AGE + EARNINGS + SIBLINGS + HEIGHT + CATGOV,
                  family = binomial,
                  data = model_data)

# Reduced models (excluding one predictor at a time)
no_AGE <- glm(MARRIED ~ EARNINGS + SIBLINGS + HEIGHT + CATGOV,
              family = binomial,
              data = model_data)

no_EARNINGS <- glm(MARRIED ~ AGE + SIBLINGS + HEIGHT + CATGOV,
                   family = binomial,
                   data = model_data)

no_SIBLINGS <- glm(MARRIED ~ AGE + EARNINGS + HEIGHT + CATGOV,
                   family = binomial,
                   data = model_data)

library(lmtest)

# Test for AGE
lrt_AGE <- lrtest(full_model, no_AGE)
# Test for EARNINGS
lrt_EARNINGS <- lrtest(full_model, no_EARNINGS)
# Test for SIBLINGS
lrt_SIBLINGS <- lrtest(full_model, no_SIBLINGS)

# Display results
lrt_AGE
lrt_EARNINGS
lrt_SIBLINGS

# turns out "EARNINGS" is not significant. "AGE" and "SIBLINGS" are significant. 
# This result makes sense, since "EARNINGS" had a high p-value in the full model.
```

## Probit vs Logit Models

### Probit Model

```{r probit_model}
model_probit <- glm(MARRIED ~ AGE + EARNINGS + SIBLINGS + HEIGHT + CATGOV,               
             data = model_data,
             family = binomial(link = "probit")) 

summary(model_probit)
```

### Logit Model

```{r logit_model}
model_logit <- glm(MARRIED ~ AGE + EARNINGS + SIBLINGS + HEIGHT + CATGOV,
             data = model_data,
             family = binomial(link = "logit")) 

summary(model_logit)
```

## Model Comparison with ROC AUC

```{r roc_auc}
# Predicted probabilities from both models
model_data$logit_prob <- predict(model_logit, type = "response")
model_data$probit_prob <- predict(model_probit, type = "response")

library(pROC)

# ROC for logit and probit
roc_logit <- roc(MARRIED ~ logit_prob, data = model_data)
roc_probit <- roc(MARRIED ~ probit_prob, data = model_data)

# Set up plot
plot(roc_logit, col = "blue", main = "ROC Curve Comparison")
lines(roc_probit, col = "red")

legend("bottomright", 
       legend = c(paste("Logit (AUC =", round(auc(roc_logit), 3), ")"), 
                 paste("Probit (AUC =", round(auc(roc_probit), 3), ")")),
       col = c("blue", "red"), lwd = 2)

abline(a = 0, b = 1, lty = 2)

# DeLong's test for AUC difference
roc_test <- roc.test(roc_logit, roc_probit)
print(roc_test)

# The test shows high p-value. Models have no significant difference in performance.
```
## threshold calculation

```{r tresholds}

# For logit model
coords_logit <- coords(roc_logit, "best", ret = "threshold", best.method = "youden")
optimal_threshold_logit <- coords_logit$threshold

# For probit model
coords_probit <- coords(roc_probit, "best", ret = "threshold", best.method = "youden")
optimal_threshold_probit <- coords_probit$threshold

# Print the optimal thresholds
print(paste("Optimal threshold for logit model:", optimal_threshold_logit))
print(paste("Optimal threshold for probit model:", optimal_threshold_probit))

# Alternatively, just type the variable names
optimal_threshold_logit
optimal_threshold_probit

```

## Treshold visualization for logit and probit

```{r treshold visualization}
# Recreate your plot with thresholds marked
plot(roc_logit, col = "blue", main = "ROC Curve Comparison with Optimal Thresholds")
lines(roc_probit, col = "red")

# Add optimal threshold points
plot(roc_logit, print.thres = optimal_threshold_logit, col = "blue", 
     print.thres.pattern = "Logit: %.3f", print.thres.cex = 0.8)
plot(roc_probit, print.thres = optimal_threshold_probit, col = "red",
     print.thres.pattern = "Probit: %.3f", print.thres.cex = 0.8)

# Add legend
legend("bottomright", 
       legend = c(paste("Logit (AUC =", round(auc(roc_logit), 3), ")"), 
                 paste("Probit (AUC =", round(auc(roc_probit), 3), ")")),
       col = c("blue", "red"), lwd = 2)

abline(a = 0, b = 1, lty = 2)
```

# Мне стало скучно, так что дальше тестим как тут будет работать питон через компилятор R

```{python matplotlib_roc}
#| fig.cap: "ROC Curves Built with Python (matplotlib)"
#| fig.width: 8
#| fig.height: 6

library(reticulate)

# Import Python modules
sklearn <- import("sklearn.metrics")
plt <- import("matplotlib.pyplot")
np <- import("numpy")

# Prepare data for Python
py_data <- model_data
py_data$MARRIED <- as.integer(py_data$MARRIED)

# Calculate ROC curves using Python
roc_logit_py <- sklearn$roc_curve(py_data$MARRIED, py_data$logit_prob)
roc_probit_py <- sklearn$roc_curve(py_data$MARRIED, py_data$probit_prob)

# Calculate AUCs
auc_logit_py <- sklearn$roc_auc_score(py_data$MARRIED, py_data$logit_prob)
auc_probit_py <- sklearn$roc_auc_score(py_data$MARRIED, py_data$probit_prob)

# Create plot
plt$figure(figsize = c(8, 6))
plt$plot(roc_logit_py[[1]], roc_logit_py[[2]], 
         color = "blue", 
         label = paste("Logit (AUC =", round(auc_logit_py, 3), ")"))
plt$plot(roc_probit_py[[1]], roc_probit_py[[2]], 
         color = "red", 
         label = paste("Probit (AUC =", round(auc_probit_py, 3), ")"))
plt$plot(c(0, 1), c(0, 1), color = "black", linestyle = "--")
plt$xlabel("False Positive Rate")
plt$ylabel("True Positive Rate")
plt$title("ROC Curves (Python Implementation)")
plt$legend(loc = "lower right")
plt$grid(TRUE)

# Display plot
plt$show()
```
