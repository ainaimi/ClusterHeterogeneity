remotes::install_github("yqzhong7/AIPW")
library(AIPW)

remotes::install_github("tlverse/tlverse")
library(tlverse)

packages <- c("data.table","tidyverse","skimr","here","sl3","mvtnorm","latex2exp","earth",
              "readxl","VGAM","ranger","xgboost","mgcv","glmnet","NbClust","factoextra",
              "SuperLearner")

for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package, repos='http://lib.stat.cmu.edu/R/CRAN')
  }
}

for (package in packages) {
  library(package, character.only=T)
}

thm <- theme_classic() +
  theme(
    legend.position = "top",
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.key = element_rect(fill = "transparent", colour = NA)
  )
theme_set(thm)

#source("./R/sl.R")

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
y0 <- 88 + 3*m[x==0] + 4*c[x==0,1] + 3*c[x==0,2] + 2*c[x==0,3] + 1*c[x==0,4] + 1*c[x==0,5] + rnorm(sum(1-x),0,20)
y1 <- 86 + 6*m[x==1] + 1*c[x==1,1] + 1*c[x==1,2] - 1.5*c[x==1,3] - 5*c[x==1,4] + 1*c[x==1,5] + rnorm(sum(x),0,15)

d0 <- data.frame(y0,m[x==0],c[x==0,],x=0);names(d0) <- c("y","m",paste0("c",1:5),"x")
d1 <- data.frame(y1,m[x==1],c[x==1,],x=1);names(d1) <- c("y","m",paste0("c",1:5),"x")

# DATA
dat <- tibble(rbind(d0,d1))
dat

## DATA GENERATION FINISHED

## DATA ANALYSIS START

## EXAMPLE 1
## simplest approach (for illustration), not recommended in practice
mod1 <- glm(y ~ x + m + x*m + c1 + c2 + c3 + c4 + c5, data=dat, family=gaussian(link="identity"))
summary(mod1)

dat1 <- transform(dat,x=1)
dat0 <- transform(dat,x=0)

head(dat1)
head(dat0)

dat$mu1 <- predict(mod1,dat1,type="response")
dat$mu0 <- predict(mod1,dat0,type="response")

dat

mean(dat$mu1-dat$mu0)

clust_dat <- dat %>% select(mu1,mu0)

clust_dat

# now search for clusters
## number of clusters via elbow method
fviz_nbclust(clust_dat, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2) +
  labs(subtitle = "Elbow method")

kmeans_res <- kmeans(clust_dat, centers = 4, nstart = 25)

plotDat <- clust_dat %>% mutate(Cluster=factor(kmeans_res$cluster))

# this figure shows the clusters identified by k-means
# results are not promising since there are clearly not four clusters
# but this toy example should not be used as test for performance of
# method
ggplot(plotDat) + 
  geom_point(aes(mu0,mu1,color=Cluster), width=.075, height=.05, size=3, shape=21) + 
  geom_abline(intercept = 0, slope = 1, linetype="dashed") +
  theme(text = element_text(size=40),
        axis.text.x = element_text(size=40),
        axis.text.y = element_text(size=40)) +
  annotate("text", label = c(TeX('$E(Y^1 - Y^0  \\; | \\; Z) = 0$')),
           x = .5, y = .6, size = 10) +
  scale_color_manual(values=c("#56B4E9","#009E73","#000000","#F0E442", "#0072B2", "#D55E00")) +
  scale_y_continuous(expand=c(0,0), lim=c(60,120)) + 
  scale_x_continuous(expand=c(0,0), lim=c(60,120))


# now regress clusters against variables in dataset to find 
# which variables explain cluster allocation
dat <- dat %>% mutate(Cluster=factor(kmeans_res$cluster))

dat

mod_clust <- vglm(Cluster ~ ., data=subset(dat,select=-c(y,mu1,mu0)),family=multinomial)
summary(mod_clust)

## EXAMPLE 2
## slightly more complex approach (for illustration)
## this approach uses random forest (via ranger) to fit the initial regression model
## instead, one should consider a meta-learner, such as super learner
## the rest is the same general process
dat <- subset(dat,select=-c(mu1,mu0,Cluster))
dat

## NB: the approaches illustrated above fits a single glm regression model to the data
## here, we accomplish same goal using 2 stratified regression models via ranger (random forest)
mod_ranger0 <- ranger(y ~ ., data=subset(dat,x==0,select=-x))
mod_ranger1 <- ranger(y ~ ., data=subset(dat,x==1,select=-x))


## When fitting a stratified regression, no need to set exposure bc the whole model 
# is generated among those who are either exposed/unexposed
# dat1 <- transform(dat,x=1)
# dat0 <- transform(dat,x=0)
# 
# head(dat1)
# head(dat0)

dat$mu1 <- predict(mod_ranger1,dat,type="response")$pred
dat$mu0 <- predict(mod_ranger0,dat,type="response")$pred

dat

mean(dat$mu1-dat$mu0)

clust_dat <- dat %>% select(mu1,mu0)

clust_dat

# now search for clusters
## number of clusters via elbow method
fviz_nbclust(clust_dat, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2) +
  labs(subtitle = "Elbow method")

kmeans_res <- kmeans(clust_dat, centers = 4, nstart = 25)

plotDat <- clust_dat %>% mutate(Cluster=factor(kmeans_res$cluster))

# this figure shows different results
# much more noise that is induced by the random forest
ggplot(plotDat) + 
  geom_point(aes(mu0,mu1,color=Cluster), width=.075, height=.05, size=3, shape=21) + 
  geom_abline(intercept = 0, slope = 1, linetype="dashed") +
  theme(text = element_text(size=40),
        axis.text.x = element_text(size=40),
        axis.text.y = element_text(size=40)) +
  annotate("text", label = c(TeX('$E(Y^1 - Y^0  \\; | \\; Z) = 0$')),
           x = .5, y = .6, size = 10) +
  scale_color_manual(values=c("#56B4E9","#009E73","#000000","#F0E442", "#0072B2", "#D55E00")) +
  scale_y_continuous(expand=c(0,0), lim=c(60,120)) + 
  scale_x_continuous(expand=c(0,0), lim=c(60,120))

dat <- dat %>% mutate(Cluster=factor(kmeans_res$cluster))

dat

mod_clust <- vglm(Cluster ~ ., data=subset(dat,select=-c(y,mu1,mu0)),family=multinomial)
summary(mod_clust)

## EXAMPLE 3 (in progress)
## refresh data
# covariates <- as.matrix(c,m)
# exposure <- as.matrix(dat$x)
# outcome <- as.matrix(dat$y)

# ## this time same approch, but instead using using AIPW + SuperLearner
# ## setting up the super learner using sl3
# sl3_list_learners("binomial") # list of available learners that can be included in SL for binary variable (in this case, exposure)
# sl3_list_learners("continuous") # list of available learners that can be included in SL for continuous variable (in this case, outcome)
# lrnr_glm <- make_learner(Lrnr_glm)
# lrnr_mean <- make_learner(Lrnr_mean)
# lrnr_ranger <- make_learner(Lrnr_ranger)
# 
# # define what goes into the SL algorithm
# sl <- Lrnr_sl$new(learners = list(lrnr_glm,lrnr_mean,lrnr_ranger))
# 
# ## fitting the AIPW estimator
# set.seed(123)
# AIPW_SL <- AIPW$new(Y = outcome,
#                     A = exposure,
#                     W = covariates, 
#                     Q.SL.library = sl,
#                     g.SL.library = sl,
#                     k_split = 3,
#                     verbose=T,
#                     save.sl.fit=T)$
#   stratified_fit()$
#   plot.p_score()$
#   plot.ip_weights()
# 
# ## AIPW Results
# print(AIPW_SL$result, digits = 2)
# 
# 
# plotDat0 <- as_tibble(list(mu0=AIPW_SL$obs_est$mu0,mu1=AIPW_SL$obs_est$mu1))



