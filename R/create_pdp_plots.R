## Matt Ryan
## Create partial dependency plots
## 15/01/2025
## Requires: tidyverse, here, glue

# libraries ---------------------------------------------------------------
pacman::p_load(tidyverse, patchwork)


# script parameters -------------------------------------------------------

text_size <- 12
height <- 12
dpi <- 600


# Functions ---------------------------------------------------------------

labels_clean <- function(labels_original){
  labels <- labels_original %>% 
    str_replace_all("_", " ") %>% 
    str_to_title() 
  labels <- str_replace(labels, " Nomask Scale", "")
  
  # labels[str_detect(labels, "State")] <- str_c(labels[str_detect(labels, "State")],
  #                                              ")")
  # labels[str_detect(labels, "State")] <- str_replace(labels[str_detect(labels,
  #                                                                      "State")],
  #                                                    "State ",
  #                                                    "State (")
  
  labels[str_detect(labels, "Gender")] <- "Gender"
  # labels[str_detect(labels, "Gender")] <- str_replace(labels[str_detect(labels,
  #                                                                       "Gender")],
  #                                                     "Gender ",
  #                                                     "Gender (")
  
  labels[str_detect(labels, "I11 Health")] <- "Willingness to isolate"
  # labels[str_detect(labels, "I11 Health")] <- str_replace(labels[str_detect(labels, "I11 Health")],
  #                                                         "I11 Health ",
  #                                                         "")
  
  # labels[str_detect(labels, "Employment")] <- str_c(labels[str_detect(labels, "Employment")],
  #                                                   ")")
  # labels[str_detect(labels, "Employment")] <- str_replace(labels[str_detect(labels,
  #                                                                           "Employment")],
  #                                                         "Employment Status ",
  #                                                         "Employment Status \n(")
  
  # labels[str_detect(labels, "I9 Health")] <- str_c(labels[str_detect(labels, "I9 Health")],
  #                                                  ")")
  labels[str_detect(labels, "I9 Health")] <- "Willingness to isolate if unwell"
  
  labels[str_detect(labels, "I2 Health")] <- "Non-household contacts"
  labels[str_detect(labels, "D1")] <- "Has commorbidities"
  
  labels[str_detect(labels, "R1 1")] <- "Perceived severity"
  labels[str_detect(labels, "R1 2")] <- "Perceived susceptibility"
  
  labels[str_detect(labels, "Phq4 1")] <- "PHQ4: Little interest or pleasure"
  labels[str_detect(labels, "Phq4 2")] <- "PHQ4: Feeling down or depressed"
  labels[str_detect(labels, "Phq4 3")] <- "PHQ4: Feeling nervous or anxious"
  labels[str_detect(labels, "Phq4 4")] <- "PHQ4: Worrying"
  
  
  labels[str_detect(labels, "Wcrex2")] <- "Confidence in government response"
  
  
  
  return(labels)
}

make_plot <- function(dat, title=""){
  
  legend_position <- "bottom"
  num_levels <- unique(dat$level)
  
  if(length(num_levels) < 2){
    title <- unique(dat$`_vname_`)
    legend_position <- "none"
  }
  
  title <- labels_clean(title)
  
  p <- dat %>% 
    ggplot(aes(x=`_x_`, y=`_yhat_`, colour=level)) +
    geom_line() +
    theme_bw() +
    labs(x = "Predictor value",
         y = "Model predictions",
         colour=NULL,
         title=title) +
    guides(color = guide_legend(nrow = 2)) +
    theme(legend.position = legend_position,
          text=element_text(size=text_size))
  
  return(p)
}

get_data <- function(model_number, model_type){
  df <- read_csv(here::here(glue::glue("results/{model_number}_{model_type}_pdp_results.csv")),
                 col_types = cols())
  df2 <- df %>% 
    mutate(variable = map_chr(str_split(`_vname_`, "_"), ~.x[1]),
           variable=case_when(str_detect(`_vname_`, "PHQ4_1") ~ "PHQ4_1",
                              str_detect(`_vname_`, "PHQ4_2") ~ "PHQ4_2",
                              str_detect(`_vname_`, "PHQ4_3") ~ "PHQ4_3",
                              str_detect(`_vname_`, "PHQ4_4") ~ "PHQ4_4",
                              str_detect(`_vname_`, "r1_1") ~ "r1_1",
                              str_detect(`_vname_`, "r1_2") ~ "r1_2",
                              str_detect(`_vname_`, "i9") ~ "i9_health",
                              str_detect(`_vname_`, "i2") ~ "i2_health",
                              str_detect(`_vname_`, "i11") ~ "i11_health",
                              str_detect(`_vname_`, "employment") ~ "employment_status",
                              TRUE ~ variable),
           level = map2_chr(`_vname_`, variable,
                            function(v_full, v){
                              split_string <- str_split(v_full, "_")[[1]]
                              if(str_detect(v, "_")){
                                ans <- split_string[3]
                                if(is.na(ans)){
                                  ans <- "1"
                                }
                              }else{
                                ans <- split_string[2]
                              }
                              if(length(split_string) < 2){
                                ans <- "1"
                              }
                              return(ans)
                            }))
  
  df_nested <- df2 %>% 
    group_by(variable) %>% 
    nest() %>% 
    mutate(plots = map2(data, variable, make_plot))
  
  return(df_nested)
}


# Make plots --------------------------------------------------------------


model_numbers <- c("model_1a",
                   "model_1b",
                   "model_2a",
                   "model_2b")
model_types <- c("rf", "xgboost")

for(num in model_numbers){
  for(tt in model_types){
    dat <- get_data(num, tt) %>% 
      arrange(variable)
    
    iter_nums <- nrow(dat)
    cc <- 0
    plot_count <- 1
    for(j in 1:ceiling(iter_nums/4)){
      tmp <- plot_count
      p <- dat$plots[[plot_count]]
      if((tmp + 1) <= iter_nums){
        for(i in (tmp+1):min(tmp + 3, iter_nums)){
          p <- p + dat$plots[[i]]
        }
        plot_count <- i + 1
        p <- p +
          plot_layout(ncol=2)
      }
      

      ggsave(here::here(glue::glue("figures/pdp_{num}_{tt}_{j}.png")),
             plot = p,
             height = height,
             width = height,
             dpi = dpi)
    }
    
    
    
  }
}

