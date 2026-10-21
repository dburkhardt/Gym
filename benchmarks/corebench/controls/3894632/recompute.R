# Evaluator-free cross-check of the source-defined unadjusted PFS Wald p value.
library(survival)
surv_dat <- read.csv('/data/clinical/survival.csv')
features <- read.csv('/data/results/reduced-feature-set.csv')
features <- features[-1]
covariates <- colnames(features)[-1]
stopifnot('Lesions.Contoured' %in% covariates[2:length(covariates)])
normalized <- as.data.frame(lapply(features, function(x) (x-mean(x))/sd(x)))
response <- Surv(surv_dat$PFS_time_RECISTv1, surv_dat$PFS_status_RECISTv1)
fit <- coxph(response ~ Lesions.Contoured, data=normalized)
value <- signif(summary(fit)$wald['pvalue'], 2)
stopifnot(length(value)==1, is.finite(value), value>=0, value<=1)
writeLines(as.character(value), '/workspace/recomputed-pfs-wald.txt')
