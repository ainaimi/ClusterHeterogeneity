remotes::install_github("yqzhong7/AIPW")
library(AIPW)

packages <- c("tidyverse","sl3","mvtnorm","ranger","SuperLearner")

for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package, repos='http://lib.stat.cmu.edu/R/CRAN')
  }
}

for (package in packages) {
  library(package, character.only=T)
}

##### https://github.com/gconzuelo/DRemm/blob/main/R/DRemm_binm.R
#####
## FUNCTION SET-UP
## TRUE VALUES
true1 <- 6
true2 <- 3

# CREATE EXPIT AND LOGIT FUNCTIONS
expit <- function(x){ exp(x)/(1+exp(x)) }
logit <- function(x){ log(x/(1-x)) }

## FUNCTION
##############################################################
#                           SET-UP                           #
##############################################################
n = 5000
p = 5
set.seed(12345)

## CONFOUNDERS (C = 5)
sigma <- matrix(0,nrow=p,ncol=p); diag(sigma) <- 1
c     <- rmvnorm(n, mean=rep(0,p), sigma=sigma)

# DESING MATRIX FOR THE OUTCOME MODEL
muMatT <- model.matrix(as.formula(paste("~(",paste("c[,",1:ncol(c),"]",collapse="+"),")")))
parms3 <- rep(3,5)
parms4 <- rep(log(1.5),5)
beta   <- parms3; beta <- c(120, beta)

# DESIGN MATRIX FOR THE PROPENSITY SCORE MODEL
piMatT  <- model.matrix(as.formula(paste("~(",paste("c[,",1:ncol(c),"]",collapse="+"),")")))
piMatT2 <- model.matrix(as.formula(paste("~(",paste("c[,",1:ncol(c),"]",collapse="+"),")")))
theta   <- c(-.5,parms4)
theta2  <- c(-.5,parms4)
mu      <- muMatT%*%beta

# PROPENSITY SCORE MODEL
pi   <- expit(piMatT%*%theta)
pi_m <- expit(piMatT2%*%theta2)
x    <- rbinom(n,1,pi)
m    <- rbinom(n,1,pi_m)

# OUTCOME MODEL: EXPOSURE VALUE UNDER M == 0 IS 6; VALUE UNDER M == 1 IS 3
y0 <- 88 + 3*m[x==0] + 4*c[x==0,1] + 3*c[x==0,2] + 2*c[x==0,3] + 1*c[x==0,4] + 1*c[x==0,5] + rnorm(sum(1-x),0,6)
y1 <- 63 + 6*m[x==1] + 1*c[x==1,1] + 1*c[x==1,2] - 1.5*c[x==1,3] - 5*c[x==1,4] + 1*c[x==1,5] + rnorm(sum(x),0,6)

d0 <- data.frame(y0,m[x==0],c[x==0,],x=0);names(d0) <- c("y","m",paste0("c",1:5),"x")
d1 <- data.frame(y1,m[x==1],c[x==1,],x=1);names(d1) <- c("y","m",paste0("c",1:5),"x")

# DATA
dat <- tibble(rbind(d0,d1))
dat

## DATA GENERATION FINISHED

## DATA ANALYSIS START

## EXAMPLE 3
## refresh data
covariates <- as.matrix(c,m)
exposure <- as.matrix(dat$x)
outcome <- as.matrix(dat$y)

## this time same approch, but instead using using AIPW + SuperLearner
## setting up the super learner using sl3
sl3_list_learners("binomial") # list of available learners that can be included in SL for binary variable (in this case, exposure)
sl3_list_learners("continuous") # list of available learners that can be included in SL for continuous variable (in this case, outcome)
lrnr_glm <- make_learner(Lrnr_glm)
lrnr_mean <- make_learner(Lrnr_mean)
lrnr_ranger <- make_learner(Lrnr_ranger)

# define what goes into the SL algorithm
sl <- Lrnr_sl$new(learners = list(lrnr_glm,lrnr_mean,lrnr_ranger))

## fitting the AIPW estimator
set.seed(123)
AIPW_SL <- AIPW$new(Y = outcome,
                    A = exposure,
                    W = covariates, 
                    Q.SL.library = sl,
                    g.SL.library = sl,
                    k_split = 3,
                    verbose=T,
                    save.sl.fit=T)$
  stratified_fit()$
  plot.p_score()$
  plot.ip_weights()

## AIPW Results
print(AIPW_SL$result, digits = 2)
