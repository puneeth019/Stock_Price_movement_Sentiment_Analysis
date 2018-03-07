library(twitteR)
library(ROAuth)
library(plyr)
library(dplyr)
library(stringr)
library(lubridate)

# set working directory
setwd(dir = "D:/DA/PGDBA/IIT/BM69006_DATA_SCIENCE_LABORATORY/dsl_project/code/")

# proxy setting to scrape data using google maps API
proxy_url <- "http://172.16.2.30:8080/"
Sys.setenv(http_proxy = proxy_url, https_proxy = proxy_url, ftp_proxy = proxy_url)

consumer_key <- "XmScX6ffN6ctcAhDCOnoO2J5i"
consumer_secret <- "M2fsCcewjox9MsAyqjsatVOqRDVAc7V2bXq5LlOW1SF4YoHg7A"

access_token <- "716695452444336129-mP4BFDigWeq1948GmPa7zmuDFGmPLZ8"
access_token_secret <- "vCAqdXddP9ci9zVc3jKf0aV2lTNfS6Fbb76LCd2fnDIA7"
setup_twitter_oauth(consumer_key,consumer_secret,access_token,access_token_secret)

googl_tweets <- searchTwitter("#GOOGL OR $GOOGL OR @Google" , n = 10000, lang = "en")
googl_df <- twListToDF(googl_tweets)

googl_df$created <- as.Date(googl_df$created)
min(googl_df$created)

googl_df1 <- googl_df[googl_df$created == min(googl_df$created),]
googl_df2 <- googl_df[googl_df$created == min(googl_df$created)+1,]
googl_df3 <- googl_df[googl_df$created == min(googl_df$created)+2,]
googl_df4 <- googl_df[googl_df$created == min(googl_df$created)+3,]
googl_df5 <- googl_df[googl_df$created == min(googl_df$created)+4,]
googl_df6 <- googl_df[googl_df$created == min(googl_df$created)+5,]
googl_df7 <- googl_df[googl_df$created == min(googl_df$created)+6,]
googl_df8 <- googl_df[googl_df$created == min(googl_df$created)+7,]
googl_df9 <- googl_df[googl_df$created == min(googl_df$created)+8,]
googl_df10 <- googl_df[googl_df$created == min(googl_df$created)+9,]
googl_df11 <- googl_df[googl_df$created == min(googl_df$created)+10,]

###############################################################################################################

#load up word polarity list and format it
afinn_list <- read.delim(file = "D:/DA/PGDBA/IIT/BM69006_DATA_SCIENCE_LABORATORY/dsl_project/data/AFINN_lexicon/imm6010/AFINN/AFINN-111.txt", 
                         header = FALSE, stringsAsFactors = FALSE)
names(afinn_list) <- c('word', 'score')
afinn_list$word <- tolower(afinn_list$word)

#categorize words as very negative to very positive and add some movie-specific words
vNegTerms <- afinn_list$word[afinn_list$score==-5 | afinn_list$score==-4]
negTerms <- c(afinn_list$word[afinn_list$score==-3 | afinn_list$score==-2 | afinn_list$score==-1])
posTerms <- c(afinn_list$word[afinn_list$score==3 | afinn_list$score==2 | afinn_list$score==1])
vPosTerms <- c(afinn_list$word[afinn_list$score==5 | afinn_list$score==4])


###################################################################################################################

#function to calculate number of words in each category within a sentence
sentimentScore <- function(sentences, vNegTerms, negTerms, posTerms, vPosTerms){
  final_scores <- matrix('', 0, 6)
  scores <- laply(sentences, function(sentence, vNegTerms, negTerms, posTerms, vPosTerms){
    initial_sentence <- sentence
    #remove unnecessary characters and split up by word 
    
    sentence <- gsub("@\\w+","",sentence)
    sentence <- gsub("#\\w+","",sentence)
    sentence <- gsub('[[:punct:]]', ' ', sentence) 
    sentence <- gsub('[[:cntrl:]]', ' ', sentence)
    sentence <- gsub('\\d+', '', sentence)
    sentence <- gsub("http\\w+","",sentence)
    
    
    #sentence <- iconv(sentence, 'UTF-8', 'ASCII')
    sentence <- tolower(sentence)
    sentence <- gsub("^rt","",sentence)
    wordList <- str_split(sentence, '\\s+')
    words <- unlist(wordList)
    #build vector with matches between sentence and each category
    vPosMatches <- match(words, vPosTerms)
    posMatches <- match(words, posTerms)
    vNegMatches <- match(words, vNegTerms)
    negMatches <- match(words, negTerms)
    #sum up number of words in each category
    vPosMatches <- sum(!is.na(vPosMatches))
    posMatches <- sum(!is.na(posMatches))
    vNegMatches <- sum(!is.na(vNegMatches))
    negMatches <- sum(!is.na(negMatches))
    
    score <- c(vNegMatches, negMatches, posMatches, vPosMatches)
    #add row to scores table
    newrow <- c(initial_sentence,sentence,score)
    final_scores <- rbind(final_scores, newrow)
    return(final_scores)
  }, vNegTerms, negTerms, posTerms, vPosTerms)
  return(scores)
}

###########################################################################################################

#build tables of positive and negative sentences with scores
googl_text <- googl_df$text
#unlist(lapply(googl_text, function(x) { str_split(x, "\n") }))

googl_text1 <- googl_df1$text
googl_text2 <- googl_df2$text
googl_text3 <- googl_df3$text
googl_text4 <- googl_df4$text
googl_text5 <- googl_df5$text
googl_text6 <- googl_df6$text
googl_text7 <- googl_df7$text
googl_text8 <- googl_df8$text
googl_text9 <- googl_df9$text
googl_text10 <- googl_df10$text
googl_text11 <- googl_df11$text





Googl.Result1 <- as.data.frame(sentimentScore(googl_text1, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result1) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result2 <- as.data.frame(sentimentScore(googl_text2, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result2) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result3 <- as.data.frame(sentimentScore(googl_text3, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result3) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result4 <- as.data.frame(sentimentScore(googl_text4, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result4) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result5 <- as.data.frame(sentimentScore(googl_text5, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result5) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result6 <- as.data.frame(sentimentScore(googl_text6, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result6) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result7 <- as.data.frame(sentimentScore(googl_text7, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result7) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result8 <- as.data.frame(sentimentScore(googl_text8, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result8) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result9 <- as.data.frame(sentimentScore(googl_text9, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result9) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result10 <- as.data.frame(sentimentScore(googl_text10, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result10) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')

Googl.Result11 <- as.data.frame(sentimentScore(googl_text11, vNegTerms, negTerms, posTerms, vPosTerms))
colnames(Googl.Result11) <- c('text','processed', 'vNeg', 'neg', 'pos', 'vPos')


Googl.Result1$vNeg <- as.integer(Googl.Result1$vNeg)-1
Googl.Result1$neg <- as.integer(Googl.Result1$neg)-1
Googl.Result1$vPos <- as.integer(Googl.Result1$vPos)-1
Googl.Result1$pos <- as.integer(Googl.Result1$pos)-1

Googl.Result1$full_score <- (Googl.Result1$vNeg*-2.5) + (Googl.Result1$neg*-1.5) + (Googl.Result1$pos*1) + (Googl.Result1$vPos*2)




Googl.Result2$vNeg <- as.integer(Googl.Result2$vNeg)-1
Googl.Result2$neg <- as.integer(Googl.Result2$neg)-1
Googl.Result2$vPos <- as.integer(Googl.Result2$vPos)-1
Googl.Result2$pos <- as.integer(Googl.Result2$pos)-1

Googl.Result2$full_score <- (Googl.Result2$vNeg*-2.5) + (Googl.Result2$neg*-1.5) + (Googl.Result2$pos*1) + (Googl.Result2$vPos*2)


Googl.Result3$vNeg <- as.integer(Googl.Result3$vNeg)-1
Googl.Result3$neg <- as.integer(Googl.Result3$neg)-1
Googl.Result3$vPos <- as.integer(Googl.Result3$vPos)-1
Googl.Result3$pos <- as.integer(Googl.Result3$pos)-1

Googl.Result3$full_score <- (Googl.Result3$vNeg*-2.5) + (Googl.Result3$neg*-1.5) + (Googl.Result3$pos*1) + (Googl.Result3$vPos*2)





Googl.Result4$vNeg <- as.integer(Googl.Result4$vNeg)-1
Googl.Result4$neg <- as.integer(Googl.Result4$neg)-1
Googl.Result4$vPos <- as.integer(Googl.Result4$vPos)-1
Googl.Result4$pos <- as.integer(Googl.Result4$pos)-1

Googl.Result4$full_score <- (Googl.Result4$vNeg*-2.5) + (Googl.Result4$neg*-1.5) + (Googl.Result4$pos*1) + (Googl.Result4$vPos*2)


Googl.Result5$vNeg <- as.integer(Googl.Result5$vNeg)-1
Googl.Result5$neg <- as.integer(Googl.Result5$neg)-1
Googl.Result5$vPos <- as.integer(Googl.Result5$vPos)-1
Googl.Result5$pos <- as.integer(Googl.Result5$pos)-1

Googl.Result5$full_score <- (Googl.Result5$vNeg*-2.5) + (Googl.Result5$neg*-1.5) + (Googl.Result5$pos*1) + (Googl.Result5$vPos*2)


Googl.Result6$vNeg = as.integer(Googl.Result6$vNeg)-1
Googl.Result6$neg = as.integer(Googl.Result6$neg)-1
Googl.Result6$vPos = as.integer(Googl.Result6$vPos)-1
Googl.Result6$pos = as.integer(Googl.Result6$pos)-1

Googl.Result6$full_score = (Googl.Result6$vNeg*-2.5) + (Googl.Result6$neg*-1.5) + (Googl.Result6$pos*1) + (Googl.Result6$vPos*2)



Googl.Result7$vNeg = as.integer(Googl.Result7$vNeg)-1
Googl.Result7$neg = as.integer(Googl.Result7$neg)-1
Googl.Result7$vPos = as.integer(Googl.Result7$vPos)-1
Googl.Result7$pos = as.integer(Googl.Result7$pos)-1

Googl.Result7$full_score = (Googl.Result7$vNeg*-2.5) + (Googl.Result7$neg*-1.5) + (Googl.Result7$pos*1) + (Googl.Result7$vPos*2)


Googl.Result8$vNeg = as.integer(Googl.Result8$vNeg)-1
Googl.Result8$neg = as.integer(Googl.Result8$neg)-1
Googl.Result8$vPos = as.integer(Googl.Result8$vPos)-1
Googl.Result8$pos = as.integer(Googl.Result8$pos)-1

Googl.Result8$full_score = (Googl.Result8$vNeg*-2.5) + (Googl.Result8$neg*-1.5) + (Googl.Result8$pos*1) + (Googl.Result8$vPos*2)


Googl.Result9$vNeg = as.integer(Googl.Result9$vNeg)-1
Googl.Result9$neg = as.integer(Googl.Result9$neg)-1
Googl.Result9$vPos = as.integer(Googl.Result9$vPos)-1
Googl.Result9$pos = as.integer(Googl.Result9$pos)-1

Googl.Result9$full_score = (Googl.Result9$vNeg*-2.5) + (Googl.Result9$neg*-1.5) + (Googl.Result9$pos*1) + (Googl.Result9$vPos*2)


Googl.Result10$vNeg = as.integer(Googl.Result10$vNeg)-1
Googl.Result10$neg = as.integer(Googl.Result10$neg)-1
Googl.Result10$vPos = as.integer(Googl.Result10$vPos)-1
Googl.Result10$pos = as.integer(Googl.Result10$pos)-1

Googl.Result10$full_score = (Googl.Result10$vNeg*-2.5) + (Googl.Result10$neg*-1.5) + (Googl.Result10$pos*1) + (Googl.Result10$vPos*2)


Googl.Result11$vNeg = as.integer(Googl.Result11$vNeg)-1
Googl.Result11$neg = as.integer(Googl.Result11$neg)-1
Googl.Result11$vPos = as.integer(Googl.Result11$vPos)-1
Googl.Result11$pos = as.integer(Googl.Result11$pos)-1

Googl.Result11$full_score = (Googl.Result11$vNeg*-2.5) + (Googl.Result11$neg*-1.5) + (Googl.Result11$pos*1) + (Googl.Result11$vPos*2)




t1 <- googl_df[,c(1,5)]
Googl.Result$text <- as.character(Googl.Result$text)
google <- left_join(Googl.Result, t1, by = "text")


