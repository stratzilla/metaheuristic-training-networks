#!/usr/bin/env Rscript

library(purrr)

# get test name through command line argument
args = commandArgs(trailingOnly=TRUE)

# get location of .csv files to use
bp_loc <- paste('../results/temp/',args[1],'-bp.csv',sep='')
ga_loc <- paste('../results/temp/',args[1],'-ga.csv',sep='')
pso_loc <- paste('../results/temp/',args[1],'-pso.csv',sep='')
de_loc <- paste('../results/temp/',args[1],'-de.csv',sep='')
ba_loc <- paste('../results/temp/',args[1],'-ba.csv',sep='')

# load the .csv files
bp <- read.csv(bp_loc, header=TRUE, sep=",")
ga <- read.csv(ga_loc, header=TRUE, sep=",")
pso <- read.csv(pso_loc, header=TRUE, sep=",")
de <- read.csv(de_loc, header=TRUE, sep=",")
ba <- read.csv(ba_loc, header=TRUE, sep=",")

phr <- 'trt' # treatment prefix

# load dataframe for Anova and Tukey
dat <- list(bp, ga, pso, de, ba) %>%
       imap_dfr(~data.frame(sample=paste0(phr,.y),value=.x[,ncol(.x)])) %>%
       lm(data=., value~sample)

# perform ANOVA test
res.aov <- anova(dat)

# perform Tukey's HSD test
res.tky <- TukeyHSD(aov(dat), conf.level=0.95)

# print results
print(res.aov)
cat("\n")
print(summary(res.aov))
cat("\n")
print(res.tky)