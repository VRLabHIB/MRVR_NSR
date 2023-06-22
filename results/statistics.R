library(psych)
library(ggpubr)


wd <- getwd()

wd <- paste0(wd , '\\Promotion\\_Mental_Rotation\\03_MR_in_VR\\MRVR_NSR\\data\\6_feature_dataset\\')

df <- read.csv2(paste0(wd, '2023-06-17_feature_dataset_agg.csv'), sep =',')

dfw <- reshape(df, idvar = "ID", timevar='dimension', direction = "wide")

dim2 <- as.numeric(dfw$Correct.2)
dim3 <- as.numeric(dfw$Correct.3)
describeBy(as.numeric(df$Correct), df$dimension)

diff <- dim2-dim3
describe(diff)
ggqqplot(diff)
shapiro.test(diff)

t.test(dim2, dim3, paired=TRUE,conf.int=TRUE) # where y1 and y2 are numeric
wilcox.test(dim2, dim3, paired=TRUE,conf.int=TRUE) # where y1 and y2 are numeric
