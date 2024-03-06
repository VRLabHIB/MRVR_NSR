setwd("C:\\Users\\Stark\\Documents\\Promotion\\_Mental_Rotation\\03_MR_in_VR\\MRVR_NSR\\data\\6_feature_dataset\\")

df <- read.csv("2023-06-17_feature_dataset_agg.csv")

dfw <- reshape(df, idvar = "ID", timevar = "dimension", direction = "wide")

result = wilcox.test(dfw$Equal.fixation.duration.within.figure.2, dfw$Equal.fixation.duration.within.figure.3, paired = TRUE)
summary(result)

diff <- dfw$Equal.fixation.duration.within.figure.2-dfw$Equal.fixation.duration.within.figure.3
median(dfw$Equal.fixation.duration.within.figure.2-dfw$Equal.fixation.duration.within.figure.3)

median(dfw$Equal.fixation.duration.within.figure.2-dfw$Equal.fixation.duration.within.figure.3)

library("plotrix")
install.packages("confintr")

print(std.error(diff))
print(median(diff))

library(effectsize)

mean(dfw$Equal.fixation.duration.within.figure.2)
rb <- rank_biserial(Pair(Equal.fixation.duration.within.figure.2, Equal.fixation.duration.within.figure.3) ~ 1, data = dfw)

library(confintr)
ci_median_diff(dfw$Equal.fixation.duration.within.figure.2,dfw$Equal.fixation.duration.within.figure.3)
