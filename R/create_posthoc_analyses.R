## Matt Ryan
## Post hoc analysis plots
## 15/08/2024
## Requires: tidyverse, here, glue


# libraries ---------------------------------------------------------------
pacman::p_load(tidyverse)


# Params ------------------------------------------------------------------
height <- 7
dpi <- 300

# Load data ---------------------------------------------------------------

model_types <- c("rf", "xgboost")
model_numbers <- c("model_1", "model_2")
features_list <- list()
cc <- 1
for(model_number in model_numbers){
  for(model_type in model_types){
    # if(model_number=="model_1"){
    #   target="face masks"
    # }else{
    #   target="general behaviour"
    # }
    
    features_list[[cc]] <- read_csv(here::here(glue::glue("results/{model_number}a_{model_type}_feature_importance.csv")),
                                    col_types = cols(), col_select = -1) %>% 
      mutate(model = model_type,
             target = model_number,
             period = "pre-mandate")
    cc <- cc+1
    features_list[[cc]] <- read_csv(here::here(glue::glue("results/{model_number}b_{model_type}_feature_importance.csv")),
                                    col_types = cols(), col_select = -1) %>% 
      mutate(model = model_type,
             target = model_number,
             period = "during mandate")
    cc <- cc+1
  }
}
feature_df <- do.call(bind_rows, features_list)

feature_summaries <- feature_df %>% 
  pivot_longer(-c(model, target, period)) %>%  
  drop_na() %>%  #protective no mask not measured in general behaviour model
  group_by(model, target, period, name) %>% 
  summarise(mean_importance = median(value),
            std = sd(value)/sqrt(n()),
            q1 = quantile(value, 0.25),
            q2 = quantile(value, 0.75),
            .groups = "drop") 

clean_data <- read_csv(here::here("data/cleaned_data.csv"),
                       col_types = cols())
clean_data_w_mandate <- read_csv(here::here("data/cleaned_data_preprocessing.csv"),
                                 col_types = cols()) %>% 
  select(RecordNo, within_mandate_period)
full_data <-left_join(clean_data, clean_data_w_mandate, by="RecordNo") %>% 
  mutate(Mandates = ifelse(within_mandate_period==0, "No", "Yes"),
         i11_health = factor(i11_health, levels = c("Not sure", "Very unwilling", "Somewhat unwilling",
                                                    "Neither willing nor unwilling",
                                                    "Somewhat willing", "Very willing")),
         i9_health = factor(i9_health, levels = c("Not sure", "No", "Yes")),
         employment_status = factor(employment_status, levels = c("Not working", "Unemployed",
                                                                  "Part time employment", 
                                                                  "Full time employment", "Retired")),
         WCRex2 = factor(WCRex2, levels = c("Don't know", "No confidence at all", "Not very much confidence",
                                            "A fair amount of confidence", "A lot of confidence")),
         WCRex1 = factor(WCRex1, levels=c("Don't know", "Very badly", "Somewhat badly",
                                          "Somewhat well", "Very well")),
         PHQ4_1 = factor(PHQ4_1, levels = c("N/A", "Prefer not to say",  "Not at all", "Several days", 
                                            "More than half the days", "Nearly every day")),
         PHQ4_2 = factor(PHQ4_2, levels = c("N/A", "Prefer not to say",  "Not at all", "Several days", 
                                            "More than half the days", "Nearly every day")),
         PHQ4_3 = factor(PHQ4_3, levels = c("N/A", "Prefer not to say",  "Not at all", "Several days", 
                                            "More than half the days", "Nearly every day")),
         PHQ4_4 = factor(PHQ4_4, levels = c("N/A", "Prefer not to say",  "Not at all", "Several days", 
                                            "More than half the days", "Nearly every day")),
         d1_comorbidities = ifelse(is.na(d1_comorbidities), "NA", d1_comorbidities),
         d1_comorbidities = str_replace_all(d1_comorbidities, "_", " "),
         d1_comorbidities = factor(d1_comorbidities, levels=c("NA", "Prefer not to say", "No", "Yes")))

# find common features ----------------------------------------------------

key_features <- feature_summaries %>% 
  group_by(model, target, period) %>% 
  arrange(-mean_importance) %>% 
  filter(!str_detect(name, "state")) %>% 
  slice(1:10) %>% 
  ungroup() %>% 
  mutate(name = ifelse(str_detect(name, "PHQ4_1"), "PHQ4_1", name),
         name = ifelse(str_detect(name, "PHQ4_2"), "PHQ4_2", name),
         name = ifelse(str_detect(name, "PHQ4_3"), "PHQ4_3", name),
         name = ifelse(str_detect(name, "PHQ4_4"), "PHQ4_4", name),
         name = ifelse(str_detect(name, "WCRex1"), "WCRex1", name),
         name = ifelse(str_detect(name, "WCRex2"), "WCRex2", name),
         name = ifelse(str_detect(name, "employment_status"), "employment_status", name),
         name = ifelse(str_detect(name, "gender"), "gender", name),
         name = ifelse(str_detect(name, "i9_health"), "i9_health", name),
         name = ifelse(str_detect(name, "i11_health"), "i11_health", name),
  ) %>% 
  distinct(target, period, name, .keep_all = TRUE) 

key_features_pre_mask <- key_features %>% 
  filter(period=="pre-mandate", target=="model_1") %>% 
  pull(name)
key_features_during_mask <- key_features %>% 
  filter(period!="pre-mandate", target=="model_1")%>% 
  pull(name)
key_features_pre_gen <- key_features %>% 
  filter(period=="pre-mandate", target!="model_1") %>% 
  pull(name)
key_features_during_gen <- key_features %>% 
  filter(period!="pre-mandate", target!="model_1")%>% 
  pull(name)


common_features_mask <- intersect(key_features_pre_mask, key_features_during_mask)

distinct_pre_mask <- key_features_pre_mask[!str_detect(key_features_pre_mask, 
                                                       str_c(common_features_mask, collapse = "|"))]
distinct_during_mask <- key_features_during_mask[!str_detect(key_features_during_mask, 
                                                             str_c(common_features_mask, collapse = "|"))]

common_features_gen <- intersect(key_features_pre_gen, key_features_during_gen)

distinct_pre_gen <- key_features_pre_gen[!str_detect(key_features_pre_gen, 
                                                     str_c(common_features_gen, collapse = "|"))]
distinct_during_gen <- key_features_during_gen[!str_detect(key_features_during_gen, 
                                                           str_c(common_features_gen, collapse = "|"))]
# post hoc analysis -------------------------------------------------------

feature_loop_list <- list(
  "masks_common" = common_features_mask,
  "gen_common" = common_features_gen,
  "masks_distinct_pre" = distinct_pre_mask,
  "masks_distinct_during" = distinct_during_mask,
  "gen_distinct_pre" = distinct_pre_gen,
  "gen_distinct_during" = distinct_during_gen
)

## Mask, commons
for(list_lab in names(feature_loop_list)){
  tmp_features <- feature_loop_list[list_lab][[1]]
  for(v in tmp_features){
    if(str_detect(list_lab, "masks")){
      tmp <- full_data %>% 
        select(target=face_mask_behaviour_binary,
               Mandates,
               var=all_of(v))
      y_label <- "Face mask usage"
    }else{
      tmp <- full_data %>% 
        select(target=protective_behaviour_binary,
               Mandates,
               var=all_of(v))
      y_label <- "Health behaviour adherence"
    }
    
    if(str_detect(list_lab, "_pre")){
      tmp <- tmp %>% 
        filter(Mandates=="No")
    }
    if(str_detect(list_lab, "_during")){
      tmp <- tmp %>% 
        filter(Mandates=="Yes")
    }
    
    
    if(isTRUE(any(is.character(tmp$var), is.factor(tmp$var)))){
      tab <- tmp %>% 
        count(target, Mandates, var) %>% 
        group_by(target, Mandates) %>% 
        mutate(p = round(n/sum(n), 4) * 100,
               n = str_c(n, " (", p, "%)")) %>% 
        select(-p) %>% 
        pivot_wider(names_from=var, values_from = n) %>% 
        arrange(Mandates)
      write_csv(tab,
                here::here(glue::glue("results/post_hoc/categorical/{list_lab}_{v}.csv")))
    }else{
      label_val <- case_when(
        v=="week_number" ~ "Survey week",
        v=="i2_health" ~ "Average number of contacts",
        v=="r1_1" ~ "Perceived severity",
        v=="r1_2" ~ "Perceived susceptibility",
        v=="cantril_ladder" ~ "Wellbeing",
        v=="household_size" ~ "Household size",
        v=="age" ~ "Age",
        TRUE ~ v
      )
      if(v=="week_number"){
        p <- tmp %>% 
          ggplot(aes(y=target, 
                     x=var,
                     fill=target)) +
          geom_violin() 
      }else{
        p <- tmp %>% 
          ggplot(aes(y=target, 
                     x=var,
                     fill=target)) +
          geom_boxplot() 
      }
      
      p <- p +
        facet_wrap(~Mandates, labeller = label_both) +
        labs(y=y_label, 
             fill=y_label, 
             x = label_val) +
        theme_bw() +
        theme(legend.position = "bottom")
      
      
      ggsave(here::here(glue::glue("results/post_hoc/continuous/{list_lab}_{v}.png")),
             plot = p,
             height = height,
             width = height,
             dpi = dpi)
    }
  }
}
