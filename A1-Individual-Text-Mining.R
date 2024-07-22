#installing and loading the mongolite library to download the Airbnb data
#install.packages("mongolite") #need to run this line of code only once and then you can comment out
library(mongolite)
library(dplyr)
library(stringr)
library(tidytext)
library(syuzhet)
library(tm)
library(cld2)
library(ggplot2)
library(topicmodels)
library(shiny)
library(tidyr)
library(wordcloud)
library(RColorBrewer)
library(reshape2)


# This is the connection_string. You can get the exact url from your MongoDB cluster screen
connection_string <- 'mongodb+srv://srage:Sumaiya15@cluster0.bixsorg.mongodb.net/'
airbnb_collection <- mongo(collection="listingsAndReviews", db="sample_airbnb", url=connection_string)

#Here's how you can download all the Airbnb data from Mongo
## keep in mind that this is huge and you need a ton of RAM memory

airbnb_all <- airbnb_collection$find()


# Data preparation: select relevant columns and filter by room type
# Assume 'room_type' and 'description' are the columns available in your dataset
room_type_data <- airbnb_all %>%
  select(room_type, description) %>%
  filter(room_type %in% c('Entire home/apt', 'Private room')) 

# Detect language and keep only English descriptions (or any other specific language)
room_type_data <- room_type_data%>%
  mutate(lang = detect_language(description))

# For the purpose of this example, we filter only English descriptions
english_descriptions <- room_type_data %>%
  filter(lang == "en")

# Text preprocessing
# Convert to lowercase, remove punctuation, replace non-ASCII characters, and trim whitespace
room_data_clean <- english_descriptions %>%
  mutate(description = tolower(description)) %>%
  mutate(description = str_replace_all(description, "[[:punct:]]", "")) %>%
  mutate(description = iconv(description, "latin1", "ASCII", sub="")) %>%
  mutate(description = str_trim(description))


###################################################################
################Text mining/analysis framework#####################


### 1. Tokenization and removal of stopwords ###
room_data_tokens <- room_data_clean %>%
  unnest_tokens(word, description) %>%
  anti_join(get_stopwords(), by = "word")

# Now i have a clean and tokenized dataset ready for analysis


# Conduct a simple word frequency analysis
word_counts <- room_data_tokens %>%
  count(room_type, word, sort = TRUE) %>%
  filter(n > 1)  # Filter to include words that appear more than once
word_counts

# Filtering for visualization to show only the top 10 words per room type
top_words_per_room_type <- word_counts %>%
  group_by(room_type) %>%
  top_n(10, n) %>%
  ungroup()

# Plotting
ggplot(top_words_per_room_type, aes(x = reorder(word, n), y = n, fill = room_type)) +
  geom_col() + # Using geom_col to create bar plot
  coord_flip() + # Flipping coordinates for better readability of word labels
  labs(x = "Frequency", y = "Words", title = "Top Words by Room Type in Airbnb Listings") +
  facet_wrap(~ room_type, scales = "free_y") +
  theme_minimal() +
  theme(legend.position = "bottom")


################################
##### 2. Sentiment Analysis#####

# AFINN sentiment analysis
afinn <- room_data_tokens %>%
  inner_join(get_sentiments("afinn")) %>%
  summarise(sentiment = sum(value)) %>%
  mutate(method = "AFINN")

# Bing sentiment analysis
bing <- room_data_tokens %>%
  inner_join(get_sentiments("bing")) %>%
  count(sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = list(n = 0)) %>%
  mutate(sentiment = positive - negative,
         method = "Bing et al.")

# Bind AFINN and Bing results
combined_sentiments <- bind_rows(afinn, bing)

# Visualize sentiments
ggplot(combined_sentiments, aes(x = method, y = sentiment, fill = method)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ method, ncol = 1, scales = "free_y")

####################################################
##### sentiment analysis based on room type ######


# AFINN sentiment analysis by room type
afinn_sentiments <- room_data_tokens %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(room_type) %>%
  summarise(sentiment = sum(value)) %>%
  ungroup() %>%
  mutate(method = "AFINN")

# Bing sentiment analysis by room type
bing_sentiments <- room_data_tokens %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  group_by(room_type, sentiment) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = sentiment, values_from = count, values_fill = list(count = 0)) %>%
  mutate(sentiment = positive - negative) %>%
  ungroup() %>%
  mutate(method = "Bing et al.")

# Combine AFINN and Bing results
combined_sentiments_by_room_type <- bind_rows(afinn_sentiments, bing_sentiments)

# Visualize sentiments by room type
ggplot(combined_sentiments_by_room_type, aes(x = room_type, y = sentiment, fill = method)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Room Type", y = "Sentiment Score", title = "Sentiment Analysis by Room Type") +
  facet_wrap(~ method, ncol = 1, scales = "free_y") +
  theme_minimal() +
  theme(legend.position = "bottom")

########################
###### 3. TF-IDF #######

# If each description is unique and corresponds to one listing, you can create a unique ID like this:
room_data_clean <- room_data_clean %>%
  mutate(description_id = dense_rank(desc(description)))

# Tokenization
room_data_tokens <- room_data_clean %>%
  unnest_tokens(word, description)

# Calculate word counts per document
word_counts <- room_data_tokens %>%
  count(description_id, word, sort = TRUE)

# Calculate the total number of documents
total_documents <- n_distinct(room_data_clean$description)

# Calculate TF-IDF
tf_idf <- word_counts %>%
  bind_tf_idf(word, description_id, n) %>%
  arrange(desc(tf_idf))

# View the results
print(tf_idf)

# Selecting top 10 terms by TF-IDF score
top_tf_idf <- tf_idf %>%
  top_n(10, tf_idf) %>%
  arrange(desc(tf_idf))

# Plotting the top TF-IDF scores
ggplot(top_tf_idf, aes(x = reorder(word, tf_idf), y = tf_idf, fill = word)) +
  geom_col(fill = "#69b3a2") +
  coord_flip() +
  labs(title = "Top 10 TF-IDF Scores in Airbnb Listings",
       x = "Terms",
       y = "TF-IDF Score") +
  theme_minimal() +
  theme(legend.position = "none")

##################################
########## 4. N-grams ############

######################
#### 4.1 Bigrams #####

# Tokenization for bigrams
bigrams <- room_data_clean %>%
  unnest_tokens(bigram, description, token = "ngrams", n = 2)

# Create a single string pattern for all stopwords
stopword_pattern <- paste(get_stopwords()$word, collapse = "|")

# Remove stopwords from bigrams
bigrams <- bigrams %>%
  mutate(bigram = str_remove_all(bigram, paste0("\\b(", stopword_pattern, ")\\b")))

# Separate the bigrams to filter out entries that are now single words or empty after removal
bigrams <- bigrams %>%
  separate(bigram, into = c("word1", "word2"), sep = " ") %>%
  filter(word1 != "", word2 != "")

# Recombine words to get the cleaned bigrams
bigrams <- bigrams %>%
  unite(bigram, word1, word2, sep = " ")

# Count the occurrence of each bigram
bigram_counts <- bigrams %>%
  count(bigram, sort = TRUE)

# View the top bigrams
top_bigrams <- bigram_counts %>%
  top_n(30, n)

# Print the top bigrams
top_bigrams

#plotting

ggplot(top_bigrams, aes(x = reorder(bigram, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(x = "Bigram", y = "Frequency", title = "Top Bigrams in Airbnb Listings") +
  theme_minimal()


######################
#### 4.2 Trigrams ####

# Tokenization for trigrams
trigrams <- room_data_clean %>%
  unnest_tokens(trigram, description, token = "ngrams", n = 3)

# Separate the trigrams to filter out entries with stopwords
trigrams <- trigrams %>%
  separate(trigram, into = c("word1", "word2", "word3"), sep = " ")

# Filter out trigrams that contain stopwords in any position
trigrams <- trigrams %>%
  filter(!word1 %in% get_stopwords()$word,
         !word2 %in% get_stopwords()$word,
         !word3 %in% get_stopwords()$word)

# Recombine words to get the cleaned trigrams
trigrams <- trigrams %>%
  unite(trigram, word1, word2, word3, sep = " ")

# Count the occurrence of each trigram
trigram_counts <- trigrams %>%
  count(trigram, sort = TRUE)

# View the top trigrams
top_trigrams <- trigram_counts %>%
  top_n(30, n)

# Print the top trigrams
top_trigrams

#plotting
ggplot(top_trigrams, aes(x = reorder(trigram, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(x = "Trigram", y = "Frequency", title = "Top Trigrams in Airbnb Listings") +
  theme_minimal()

#######################
####### 5. LDA ########

# Create a document-term matrix, which is needed for LDA
dtm <- room_data_clean %>%
  unnest_tokens(word, description) %>%
  anti_join(get_stopwords(), by = "word") %>%  # Remove stopwords
  count(document = row_number(), word) %>%
  cast_dtm(document, word, n)

# Fit the LDA model
# You can change the number of topics based on your assessment of the data
lda_model <- LDA(dtm, k = 5, control = list(seed = 1234))

# Examine the topics
topics <- tidy(lda_model, matrix = "beta")

# Get top terms for each topic
top_terms <- topics %>%
  group_by(topic) %>%
  #top_n(140, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# View top terms in each topic
print(top_terms, n=500)


#plotting
# First, select the top 10 terms for each topic based on the highest beta values
top_terms_filtered <- top_terms %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, desc(beta))

# Now, plot the results
ggplot(top_terms_filtered, aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  labs(x = "Term", y = "Beta", title = "Top Terms in Each Topic from LDA Model")


#############################
####### 6. ZIPFS laws #######

# Calculate the frequency of each word
word_freq <- room_data_tokens %>%
  count(word, sort = TRUE) %>%
  ungroup()

# Add ranks to the data frame
word_freq <- word_freq %>%
  mutate(rank = row_number(), 
         freq = n/sum(n))

# Plotting Zipf's law
ggplot(word_freq, aes(x = rank, y = freq)) +
  geom_line() +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Rank", y = "Frequency", 
       title = "Word Frequency vs. Rank (Zipf's Law)") +
  theme_minimal()

##########################################
########## Numerical analysis ############

# Retrieve data and add a unique identifier right away
airbnb_numerical <- airbnb_collection$find() %>%
  select(room_type, description, price, number_of_reviews) %>%
  filter(room_type %in% c('Entire home/apt', 'Private room')) %>%
  mutate(description_id = row_number())  # Create a unique ID for each row right after data retrieval

# Normalize 'price' and continue with data cleaning
airbnb_numerical <- airbnb_numerical %>%
  mutate(price = (price - min(price, na.rm = TRUE)) / (max(price, na.rm = TRUE) - min(price, na.rm = TRUE))) %>%
  mutate(lang = detect_language(description)) %>%
  filter(lang == "en") %>%
  mutate(description = tolower(description)) %>%
  mutate(description = str_replace_all(description, "[[:punct:]]", "")) %>%
  mutate(description = iconv(description, "latin1", "ASCII", sub="")) %>%
  mutate(description = str_trim(description))


# Calculate sentiment scores using the AFINN lexicon
airbnb_numerical_clean <- airbnb_numerical %>%
  unnest_tokens(word, description) %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(description_id) %>%
  summarise(sentiment_score = sum(value)) %>%
  ungroup()

# Merge sentiment data back with the main numerical data frame
listing_data <- airbnb_numerical %>%
  select(description_id, room_type, price, number_of_reviews) %>%
  left_join(airbnb_numerical_clean, by = "description_id")


################################
#####Correlation Analysis#######

# Calculate correlations between numerical data and sentiment scores
correlation_matrix <- cor(listing_data[, c("price", "number_of_reviews", "sentiment_score")], use = "complete.obs")

# Print the correlation matrix to view the correlation coefficients
print(correlation_matrix)

# Melt the correlation matrix for visualization
melted_correlation_matrix <- melt(correlation_matrix, varnames = c("Variable1", "Variable2"))

# Create a heatmap of the correlations
ggplot(melted_correlation_matrix, aes(x = Variable1, y = Variable2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "navy", high = "skyblue", midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson Correlation") +
  geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 4) +
  labs(title = "Heatmap of Correlation Matrix", x = "", y = "") +
  theme_minimal()



###################
#####R shiny####### 

# Define UI for application
ui <- fluidPage(
  # Application title
  titlePanel("Airbnb Text Analysis Dashboard"),
  
  # Sidebar with a tabset that includes each analysis
  sidebarLayout(
    sidebarPanel(
      h3("Text Analysis Dashboard"),
      helpText("This dashboard contains multiple tabs, each displaying different types of analysis results.")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Frequency Analysis", plotOutput("wordFreqPlot")),
        tabPanel("Sentiment Analysis", plotOutput("sentimentPlot")),
        tabPanel("TF-IDF Analysis", plotOutput("tfidfPlot")),
        tabPanel("N-Grams Analysis", plotOutput("ngramsPlot")),
        tabPanel("Topic Modeling", plotOutput("ldaPlot")),
        tabPanel("Zipf's Law", plotOutput("zipfPlot")),
        tabPanel("Correlation Heatmap", plotOutput("correlationPlot"))
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  output$wordFreqPlot <- renderPlot({
    ggplot(top_words_per_room_type, aes(x = reorder(word, n), y = n, fill = room_type)) +
      geom_col() + # Using geom_col to create bar plot
      coord_flip() + # Flipping coordinates for better readability of word labels
      labs(x = "Frequency", y = "Words", title = "Top Words by Room Type in Airbnb Listings") +
      facet_wrap(~ room_type, scales = "free_y") +
      theme_minimal() +
      theme(legend.position = "bottom")
  })
  
  output$sentimentPlot <- renderPlot({
    ggplot(combined_sentiments, aes(x = method, y = sentiment, fill = method)) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~ method, ncol = 1, scales = "free_y")
  })
  
  output$tfidfPlot <- renderPlot({
    ggplot(top_tf_idf, aes(x = reorder(word, tf_idf), y = tf_idf, fill = word)) +
      geom_col(fill = "#69b3a2") +
      coord_flip() +
      labs(title = "Top 10 TF-IDF Scores in Airbnb Listings",
           x = "Terms",
           y = "TF-IDF Score") +
      theme_minimal() +
      theme(legend.position = "none")
  })
  
  output$ngramsPlot <- renderPlot({
    ggplot(top_bigrams, aes(x = reorder(bigram, n), y = n)) +
      geom_col() +
      coord_flip() +
      labs(x = "Bigram", y = "Frequency", title = "Top Bigrams in Airbnb Listings") +
      theme_minimal()
  })
  
  output$ldaPlot <- renderPlot({
    ggplot(top_terms_filtered, aes(term, beta, fill = factor(topic))) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~ topic, scales = "free_y") +
      coord_flip() +
      labs(x = "Term", y = "Beta", title = "Top Terms in Each Topic from LDA Model")
  })
  
  output$zipfPlot <- renderPlot({
    ggplot(word_freq, aes(x = rank, y = freq)) +
      geom_line() +
      scale_x_log10() +
      scale_y_log10() +
      labs(x = "Rank", y = "Frequency", 
           title = "Word Frequency vs. Rank (Zipf's Law)") +
      theme_minimal()
  })
  
  output$correlationPlot <- renderPlot({
    ggplot(melted_correlation_matrix, aes(x = Variable1, y = Variable2, fill = value)) +
      geom_tile() +
      scale_fill_gradient2(low = "navy", high = "skyblue", midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson Correlation") +
      geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 4) +
      labs(title = "Heatmap of Correlation Matrix", x = "", y = "") +
      theme_minimal()
  })
}

# Run the application
shinyApp(ui = ui, server = server)