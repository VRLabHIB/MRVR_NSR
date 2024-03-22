library(psych)
library(ggpubr)
library(lavaan)
library(dplyr)
library(jmv)

wd <- getwd()

wd <- paste0(wd , '\\Promotion\\_Mental_Rotation\\03_MR_in_VR\\MRVR_NSR\\data\\6_feature_dataset\\')

temp <- paste0(wd ,'temp\\')

df <- read.csv2(paste0(wd, '2024-03-21_final_feature_dataset_agg.csv'), sep =',')

df$dimension = as.numeric(df$dimension)

dfw <- reshape(df, idvar = "ID", timevar='dimension', direction = "wide")


#Describptives

dfw$Correct.2 <- as.numeric(dfw$Correct.2)
dfw$Correct.3 <- as.numeric(dfw$Correct.3)

s2 <- describe(dfw$Correct.2)
s2 <- cbind(Variable = "Correct 2D", s2)

s3 <- describe(dfw$Correct.3)
s3 <- cbind(Variable = "Correct 3D", s3)

c <- rbind(s2, s3)
f <- c

#################
dfw$RT.2 <- as.numeric(dfw$RT.2)
dfw$RT.3 <- as.numeric(dfw$RT.3)

s2 <- describe(dfw$RT.2)
s2 <- cbind(Variable = "RT 2D", s2)

s3 <- describe(dfw$RT.3)
s3 <- cbind(Variable = "RT 3D", s3)

c <- rbind(s2, s3)
f <- rbind(f,c)


#################
var2 <- as.numeric(dfw$Mean.fixation.duration.2)
var3 <- as.numeric(dfw$Mean.fixation.duration.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean fixation duration 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean fixation duration 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Mean.saccade.rate.2)
var3 <- as.numeric(dfw$Mean.saccade.rate.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean fixation rate 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean fixation rate 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Mean.regressive.fixation.duration.2)
var3 <- as.numeric(dfw$Mean.regressive.fixation.duration.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean regressive fixation duration 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean regressive fixation duration 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Equal.fixation.duration.between.figures.2)
var3 <- as.numeric(dfw$Equal.fixation.duration.between.figures.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Equal fixation duration between figure 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Equal fixation duration between figure 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)


#################
var2 <- as.numeric(dfw$Equal.fixation.duration.within.figure.2)
var3 <- as.numeric(dfw$Equal.fixation.duration.within.figure.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Equal fixation duration within figures 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Equal fixation duration within figures 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Strategy.ratio.2)
var3 <- as.numeric(dfw$Strategy.ratio.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Strategy ratio 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Strategy ratio 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)


#################
var2 <- as.numeric(dfw$Mean.saccade.velocity.2)
var3 <- as.numeric(dfw$Mean.saccade.velocity.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean saccade velocity 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean saccade velocity 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Mean.saccade.rate.2)
var3 <- as.numeric(dfw$Mean.saccade.rate.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean saccades rate 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean saccades rate 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Mean.pupil.diameter.2)
var3 <- as.numeric(dfw$Mean.pupil.diameter.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean pupil diameter 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean pupil diameter 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Pupil.diameter.amplitude.2)
var3 <- as.numeric(dfw$Pupil.diameter.amplitude.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Peak pupil diameter 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Peak pupil diameter 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Mean.head.rotation.2)
var3 <- as.numeric(dfw$Mean.head.rotation.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean head rotation 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean head rotation 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Mean.head.movement.2)
var3 <- as.numeric(dfw$Mean.head.movement.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean head movement 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean head movement 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)

#################
var2 <- as.numeric(dfw$Mean.distance.to.figure.2)
var3 <- as.numeric(dfw$Mean.distance.to.figure.3)

s2 <- describe(var2)
s2 <- cbind(Variable = "Mean distance to figure 2D", s2)

s3 <- describe(var3)
s3 <- cbind(Variable = "Mean distance to figure 3D", s3)

c <- rbind(s2, s3)

f <- rbind(f,c)


f <- f %>% 
  mutate_if(is.numeric, round, digits = 3)

drop <- c('vars', 'n', 'trimmed', 'range' )
ff <- f[,!(names(f) %in% drop)]

write.csv(ff, paste0(temp , 'stat1.csv'))

#library(Hmisc)
#latex(ff, file="", multicol=FALSE, rowlabel.just='l', rowname=NULL) 

###############################################################################
drop <- c('X.2', 'X.3')
dfww <- dfw[,!(names(dfw) %in% drop)]
dfww <- as.numeric(dfww)

dfww <- as.data.frame(sapply(dfww, as.numeric))

cols <- colnames(df)[c(-1)][c(-1)]


results <- ttestPS(
  data = dfww,
  pairs = list(
    list(
      i1 = 'Correct.2',
      i2 = 'Correct.3')),
  wilcoxon = TRUE,
  meanDiff = TRUE,
  effectSize = TRUE,
  ci = TRUE,
  desc = TRUE, 
  norm = TRUE)

x <- data.frame(results$ttest)

for(col in cols[c(-1)]){
  results <- ttestPS(
    data = dfww,
    pairs = list(
      list(
        i1 = paste0(col, '.2'),
        i2 = paste0(col, '.3'))),
    wilcoxon = TRUE,
    meanDiff = TRUE,
    effectSize = TRUE,
    ci = TRUE,
    desc = TRUE, 
    norm = TRUE)  

  x1 <-  data.frame(results$ttest)
  x <- rbind(x, x1)
  
}
  
write.csv(x, paste0(temp, 'stat2.csv'))

####################################################################
r <- results$ttest
r <- r %>% 
  mutate_if(is.numeric, round, digits = 3)
r

t <- results$ttest$asDF
t <- t %>% 
  mutate_if(is.numeric, round, digits = 3)
t[,c(0:12)]
t[,c(13:23)]

results$norm


