## Matt Ryan
## Extract important features to make partial dependency plots
## 15/01/2025
## Requires: tidyverse, here, glue

# libraries ---------------------------------------------------------------
pacman::p_load(tidyverse)


# Functions ---------------------------------------------------------------

get_important_features <- function(model_number, model_type){
  df <- read_csv(here::here(glue::glue("results/{model_number}_{model_type}_feature_importance.csv")),
                  col_types = cols())
  
  df_long <- df %>% 
    pivot_longer(-1) %>% 
    group_by(name) %>% 
    summarise(mean_importance = median(value)) %>% 
    arrange(mean_importance)
  
  feat_list <- df_long %>% 
    filter(!str_detect(name, "state")) %>% 
    slice_max(n=10, order_by = mean_importance) %>% 
    pull(name)
  feat_types <- map_chr(str_split(feat_list, "_"), ~.x[1])
  
  
  for(k in 1:length(feat_types)){
    if(str_detect(feat_types[k], "PHQ")){
      feat_types[k] = str_c(str_split(feat_list[k], "_")[[1]][1:2], collapse="_")
    }
  }
  
  feat_find <- str_c(c(feat_types, "state"), collapse ="|")
  
  final_features <- df_long %>% 
    filter(str_detect(name, feat_find)) %>% 
    pull(name)
  
  res <- tibble(features=final_features) %>% 
    mutate(index=row_number()) %>% 
    select(index, features)
  
  return(res)
}


# get data ----------------------------------------------------------------

model_numbers <- c("model_1a",
                 "model_1b",
                 "model_2a",
                 "model_2b")
model_types <- c("rf", "xgboost")

for(num in model_numbers){
  for(tt in model_types){
    ans <- get_important_features(num, tt)
    write_csv(x = ans, here::here(glue::glue("results/{num}_{tt}_top_features.csv")))
  }
}
