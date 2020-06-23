---
#CAPSTONE PROJECT – Hotel Bookings Cancellation Prediction  

  ####by:reena2930   

#Overview/Introduction    

ref: https://www.kaggle.com/jessemostipak/hotel-booking-demand   

***PLEASE ATTACH THE SUPPORTING "hotel_bookingsCSV" file WHEN THE SYSTEM PROMPTS FOR THE CODING TO WORK***
# Note: This process could take a couple of minutes if you do not have the packages updated.          

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

#PLEASE ATTACH THE SUPPORTING hotel_bookingsCSV file WHEN THE SYSTEM PROMPTS FOR THE CODING TO WORK.

hotel_bookings<-read.csv(file.choose(), header=T)

head(hotel_bookings)

#removing unwanted columns from the dataset
drop_cols = c("arrival_date_week_number", "is_repeated_guest", 
              "previous_bookings_not_canceled", "adults", "children", 
              "babies", "stays_in_weekend_nights", "stays_in_week_nights",
              "agent", "company", "days_in_waiting_list", "deposit_type", 
              "total_of_special_requests", "reservation_status", 
              "reservation_status_date", "country", 
              "required_car_parking_spaces")

hoteldat<-hotel_bookings%>%select(-drop_cols)

#dataset with revised number of columns
head(hoteldat)

#using as_factor to convert variables into a factor to preserve the value and variable label attributes
hoteldat <- hoteldat %>%
  mutate(hotel = as_factor(hotel),
         is_canceled = as_factor(is_canceled),
         arrival_date_year = as_factor(arrival_date_year),
         arrival_date_month = as_factor(arrival_date_month),
         meal = as_factor(meal),
         market_segment = as_factor(market_segment),
         distribution_channel = as_factor(distribution_channel),
         previous_cancellations = as_factor(previous_cancellations),
         reserved_room_type = as_factor(reserved_room_type),
         assigned_room_type = as_factor(assigned_room_type)) 

#changing the 0 to No and 1 to Yes, for visualization purpose
hoteldat<-hoteldat%>%
  mutate(is_canceled = ifelse(str_detect(is_canceled,"0")==TRUE,"No","Yes"))

summary(hoteldat)

head(hoteldat)

dim(hoteldat)

         
set.seed(1, sample.kind="Rounding")
#we will use a data partition set with 80% training data and 20% test data
test_index <- createDataPartition(y = hoteldat$is_canceled, times = 1, p = 0.2, list = FALSE)
train_set<-hoteldat[-test_index,]
test_set<-hoteldat[test_index,]

#Data Exploration       

summary(train_set)

train_set%>%group_by(is_canceled)%>%summarize(n=n())

head(train_set)

#Proportion of bookings that were canceled
train_set%>%ggplot(aes(x=hotel, fill=is_canceled))+geom_bar()+ggtitle("Canceled Booking by Hotel")+geom_text(stat = "count", aes(label=..count..),vjust=-0.05)

#Bookings canceled between 2015-2017  
train_set%>%ggplot(aes(x=arrival_date_year, fill=is_canceled))+geom_bar()+ggtitle("Canceled Bookings Trend by Years")+geom_text(stat = "count", aes(label=..count..),vjust=-0.05)
  
###Monthly Booking cancellations    
train_set%>%ggplot(aes(x=arrival_date_month, fill=is_canceled))+geom_bar()+coord_flip()+ggtitle("Canceled Booking by Months")+geom_text(stat = "count", aes(label=..count..),vjust=-0.05)

#Bookings canceled based lead time  
train_set%>%ggplot(aes(x=lead_time, fill=is_canceled))+geom_bar()+ggtitle("Canceled Booking by Lead Time")+geom_text(stat = "count", aes(label=..count..),vjust=-0.05)

#Bookings canceled based Market segment   
train_set%>%ggplot(aes(x=market_segment, fill=is_canceled))+geom_bar()+coord_flip()+ggtitle("Canceled Booking by Market Segment")

#Bookings canceled based on Previous Cancellations   
train_set%>%ggplot(aes(x=previous_cancellations, fill=is_canceled))+geom_bar()+ggtitle("Canceled Bookings by Previous Cancellations")

#Model Preparation
#LDA model

#LDA model on train set
set.seed(1, sample.kind = "Rounding")

train_lda <- train(is_canceled ~ arrival_date_year + lead_time + adr + arrival_date_month + booking_changes, method = "lda", data = train_set)
lda_preds <- predict(train_lda, train_set)
confusionMatrix(data=lda_preds, reference=factor(train_set$is_canceled))$overall["Accuracy"]


#LDA model on test set
test_lda <- train(is_canceled ~ arrival_date_year + lead_time + adr + arrival_date_month + booking_changes, method = "lda", data = train_set)
lda_preds <- predict(test_lda, test_set)
confusionMatrix(data=lda_preds, reference=factor(test_set$is_canceled))$overall["Accuracy"]


#Logistic regression model

#1st glm model on train set
set.seed(1, sample.kind = "Rounding")
train_glm<- train(is_canceled ~ arrival_date_year + lead_time + adr, method = "glm", data = train_set)
glm_pred <- predict(train_glm, train_set)
confusionMatrix(data=glm_pred, reference=factor(train_set$is_canceled))$overall["Accuracy"]

#1st glm model on test set
test_glm<- train(is_canceled ~ arrival_date_year + lead_time + adr, 
method = "glm", data = train_set)
glm_pred <- predict(test_glm, test_set)
confusionMatrix(data=glm_pred, reference=factor(test_set$is_canceled))$overall["Accuracy"]

#2nd glm model on train set
set.seed(1, sample.kind = "Rounding")
train_glm2<- train(is_canceled ~ arrival_date_year + lead_time + adr + arrival_date_month + booking_changes, method = "glm", data = train_set)
glm_pred2 <- predict(train_glm2, train_set)
confusionMatrix(data=glm_pred2, reference=factor(train_set$is_canceled))$overall["Accuracy"]

#2nd glm model on test set
test_glm2<- train(is_canceled ~ arrival_date_year + lead_time + adr + arrival_date_month + booking_changes, method = "glm", data = train_set)
glm_pred2<- predict(test_glm2, test_set)
confusionMatrix(data=glm_pred2, reference=factor(test_set$is_canceled))$overall["Accuracy"]



