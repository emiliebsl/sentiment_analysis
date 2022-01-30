library(ggplot2)
library(quanteda)
library(quanteda.textmodels)
library(utf8)
library(caret)
library(glmnet)

##Loading and view of data
tweets <- read.csv("tweets.csv", header = T)
str(tweets)
View(tweets)
summary(tweets)

##Basic checks
lines <- readLines("tweets.csv", encoding = "UTF-8")
lines[!utf8_valid(lines)]
#All lines are made of correct UTF-8 characters
linesQ_NFC <- utf8_normalize(lines)
sum(linesQ_NFC != lines)
#The text is in NFC.

#Barplot
racistsexist <- length(which(tweets$label == 1))
neutral <- length(which(tweets$label == 0))
Sentiment <- c("Racist/Sexist","Neutral")
Count <- c(racistsexist, neutral)
output <- data.frame(Sentiment,Count)
output$Sentiment<-factor(output$Sentiment,levels=Sentiment)
ggplot(output, aes(x=Sentiment,y=Count))+
  geom_bar(stat = "identity", aes(fill = Sentiment))+
  ggtitle("Barplot of the total number of tweets per label")

#Tokenize text
corp <- corpus(tweets, text_field = "tweet")
summary(corp, 5)
head(docvars(corp))
toks <- tokens(corp, remove_punct = TRUE, remove_number = TRUE, remove_symbols = TRUE, what = "word1")
toks <- tokens_remove(toks, pattern = c("user", stopwords("en")))
toks %>% tokens_split(separator = "'",valuetype = c("fixed", "regex"),remove_separator = TRUE)

#Document frequency matrix
dfmat <- dfm(toks)
dfmat <- dfm_keep(dfmat, min_nchar = 3) 
dfmat <- dfm_trim(dfmat, max_docfreq = 0.1, docfreq_type = "prop")
topfeatures(dfmat, 10)

#Get training and data sets
set.seed(1)
id_train <- sample(1:31962, 25570, replace = FALSE)
dfmat_training <- dfm_subset(dfmat, id %in% id_train)
dfmat_test <- dfm_subset(dfmat, !id %in% id_train)

#Naive Bayes Classifier
tmod_nb <- textmodel_nb(dfmat_training, dfmat_training$label)
dfmat_matched <- dfm_match(dfmat_test, features = featnames(dfmat_training))
class <- dfmat_matched$label
predicted_class <- predict(tmod_nb, newdata = dfmat_matched)
tab_class <- table(class, predicted_class)
tab_class
confusionMatrix(tab_class, positive = "1", mode = "everything")

#Regularized regression classifier
lasso <- cv.glmnet(x = dfmat_training,
                   y = as.integer(dfmat_training$label == "1"),
                   alpha = 1,
                   nfold = 5,
                   family = "binomial")

index_best <- which(lasso$lambda == lasso$lambda.min)
beta <- lasso$glmnet.fit$beta[, index_best]
dfmat_matched <- dfm_match(dfmat_test, features = featnames(dfmat_training))
pred <- predict(lasso, dfmat_matched, type = "response", s = lasso$lambda.min)
class <- as.integer(dfmat_matched$label == "1")
predicted_class <- as.integer(predict(lasso, dfmat_matched, type = "class"))
tab_class <- table(class, predicted_class)
tab_class
confusionMatrix(tab_class, positive = "1", mode = "everything")
