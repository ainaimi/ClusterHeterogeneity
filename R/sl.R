##############################################################
#                        SuperLearner                        #
##############################################################
## SUPER LEARNER LIBRARY
## LOADING ALL
ranger_learner1  <- create.Learner("SL.ranger", tune=list(min.node.size=30, num.trees=500,max.depth=2))
ranger_learner2  <- create.Learner("SL.ranger", tune=list(min.node.size=30, num.trees=500,max.depth=3))
glmnet_learner   <- create.Learner("SL.glmnet", tune=list(alpha = seq(0,1,.2)))
earth_learner1   <- create.Learner("SL.earth",tune=list(degree=2))
earth_learner2   <- create.Learner("SL.earth",tune=list(degree=3))
svm_learner     <- create.Learner("SL.svm",tune=list(nu = c(.25,.5,.75),degree=c(3,4)))
gam_learner1     <- create.Learner("SL.gam",tune=list(deg.gam=2))
gam_learner2     <- create.Learner("SL.gam",tune=list(deg.gam=3))
gam_learner3     <- create.Learner("SL.gam",tune=list(deg.gam=4))
SL.library <- c("SL.glm","SL.nnet","SL.mean","SL.bayesglm", ranger_learner1$names, ranger_learner1$names,
                glmnet_learner$names, earth_learner1$names, earth_learner2$names, svm_learner$names,
                gam_learner1$names, gam_learner2$names, gam_learner3$names)

## FINAL LIBRARY
sl.lib_mu <- sl.lib_pi <- SL.library

##############################################################
#                            SL 3                            #
##############################################################
ranger_lrn1 <- make_learner(Lrnr_ranger, min.node.size = 30, num.trees=500, max.depth=2)
ranger_lrn2 <- make_learner(Lrnr_ranger, min.node.size = 30, num.trees=500, max.depth=3)
glmnet_lrn  <- make_learner(Lrnr_glmnet, alpha = seq(0,1,.2))
earth_lrn1  <- make_learner(Lrnr_earth, degree = 2)
earth_lrn2  <- make_learner(Lrnr_earth, degree = 3)
bglm_lrnr <- Lrnr_pkg_SuperLearner$new("SL.bayesglm")
svm_lrnr  <- Lrnr_pkg_SuperLearner$new("SL.svm")
gam_lrnr   <- Lrnr_pkg_SuperLearner$new("SL.gam")
glm_lrnr   <- Lrnr_pkg_SuperLearner$new("SL.glm")
mean_lrnr  <- Lrnr_pkg_SuperLearner$new("SL.mean")
nnet_lrnr  <- Lrnr_pkg_SuperLearner$new("SL.nnet")

# create a list with all learners
lrn_list <- list(ranger_lrn1, ranger_lrn2, glmnet_lrn, earth_lrn1, earth_lrn2, bglm_lrnr, svm_lrnr,
                 gam_lrnr, glm_lrnr, mean_lrnr, nnet_lrnr)

# define metalearners appropriate to data types
metalearner <- make_learner(Lrnr_nnls)

# Define the sl_Y and sl_A (we only need 1 because both are same type)
sl_lib <- Lrnr_sl$new(learners = lrn_list, metalearner = make_learner(Lrnr_nnls))