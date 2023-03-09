---
title: "TM Exam1"
author: "Chanthrika Palanisamy"
date: "2023-03-09"
output: html_document
---

knitr::opts_chunk$set(echo = TRUE)


# Clustering - Hierarchical

# Create the corpus manually and now load them

library(corpus)
library(tm)

SmallCorpus <- Corpus(DirSource("/Users/rika/Documents/TM/exam1/corpus"))
ndocs<-length(SmallCorpus)

## Do some clean-up.............
SmallCorpus <- tm_map(SmallCorpus, content_transformer(tolower))
SmallCorpus <- tm_map(SmallCorpus, removePunctuation)
## Remove all Stop Words
SmallCorpus <- tm_map(SmallCorpus, removeWords, stopwords("english"))



# install.packages("tm")
library(tm)

# Convert the data into to a Document Term Matrix
# hclust_in_data<-data$content

SmallCorpus_DTM <- DocumentTermMatrix(SmallCorpus,
                                 control = list(
                                   stopwords = TRUE, ## remove normal stopwords
                                   wordLengths=c(3, 10), ## get rid of words of len 2 or smaller or larger than15
                                   removePunctuation = TRUE,
                                   removeNumbers = TRUE,
                                   tolower=TRUE
                                 ))

inspect(SmallCorpus_DTM)

#silohuette

# install.packages("NbClust")
library(NbClust)
library(factoextra)
# Convert to DF
SmallCorpus_DF_DT <- as.data.frame(as.matrix(SmallCorpus_DTM))

# Using Sihouette to determine the optimal number of clusters
fviz_nbclust(SmallCorpus_DF_DT, method = "silhouette", FUN = hcut, k.max = 9)

#Source: http://www.sthda.com/english/articles/29-cluster-validation-essentials/96-determiningthe-optimal-number-of-clusters-3-must-know-methods/#:~:text=fviz_nbclust()%20function%20%5Bin%20factoextra,)%2C%20CLARA%2C%20HCUT%5D




#Cosine similarity and Hclust

(My_m <- (as.matrix(scale(t(SmallCorpus_DF_DT)))))
(My_cosine_dist = 1-crossprod(My_m) /(sqrt(colSums(My_m^2)%*%t(colSums(My_m^2)))))
# create dist object
My_cosine_dist <- as.dist(My_cosine_dist) ## Important
HClust_Ward_CosSim_SmallCorp2 <- hclust(My_cosine_dist, method="ward.D")
plot(HClust_Ward_CosSim_SmallCorp2, cex=.7, hang=-30,main = "Cosine Sim")
rect.hclust(HClust_Ward_CosSim_SmallCorp2, k=3)


#ARM
#detach(package:tm, unload=TRUE)

library(arules)
library(tm)

#loading the dataset by setting the all StringsAsFactor = TRUE
arm <- read.transactions("/Users/rika/Documents/TM/exam1/exam1_transaction_nolabel.csv",  rm.duplicates = TRUE, format = "basket", sep = ",")

#loading the dataset by setting the all StringsAsFactor = TRUE
arm_l <- read.transactions("/Users/rika/Documents/TM/exam1/exam1_transaction.csv",  rm.duplicates = TRUE, format = "basket", sep = ",")

#Setting support, confidence and calling apriori from arules package
rules = arules::apriori(arm, parameter = list(support=.0133, 
                                                 confidence=0.5, minlen=2))

sup_rules <- sort(rules, decreasing=TRUE, by="support")
sup_rules <- sup_rules[!is.redundant(sup_rules)]
conf_rules <- sort(rules, decreasing=TRUE, by="confidence")
conf_rules <- conf_rules[!is.redundant(conf_rules)]
lift_rules <- sort(rules, decreasing=TRUE, by="lift")
lift_rules <- lift_rules[!is.redundant(lift_rules)]

gi <- generatingItemsets(sup_rules)
d <- which(duplicated(gi))
sup_rules = sup_rules[-d]
gi <- generatingItemsets(conf_rules)
d <- which(duplicated(gi))
conf_rules = conf_rules[-d]
gi <- generatingItemsets(lift_rules)
d <- which(duplicated(gi))
lift_rules = lift_rules[-d]

rules <- rules[!is.redundant(rules)]
gi <- generatingItemsets(rules)
d <- which(duplicated(gi))
rules = rules[-d]

arules::inspect(rules)

inspect(lift_rules[1:71])
inspect(sup_rules[1:71])
inspect(conf_rules[1:71])

#Visualizing rules
subrules <- head(sort(rules, by="confidence"),15)
plot(subrules, method="graph", engine="htmlwidget")

#Setting support, confidence, RHS = "" and calling apriori from arules package
polkadot_Rules <- arules::apriori(data=arm_l,parameter = list(supp=.01, conf=.5, minlen=2),
                     appearance = list(default="lhs", rhs="polkadot"),
                     control=list(verbose=FALSE))


polkadot_Rules <- polkadot_Rules[!is.redundant(polkadot_Rules)]
conf_polkadot_Rules <- sort(polkadot_Rules, decreasing=TRUE, by="support")
inspect(conf_polkadot_Rules[1:29])

#Visualizing rules when RHS = "polkadot"
plot(polkadot_Rules, method="graph", engine="htmlwidget")









