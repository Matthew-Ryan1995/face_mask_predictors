## Matt Ryan
## Create feature importance plots
## 05/08/2024
## Requires: tidyverse, here, glue
#TODO: Add group to mean plot, colour by it

# libraries ---------------------------------------------------------------
pacman::p_load(tidyverse, patchwork)


# script parameters -------------------------------------------------------

text_size <- 16
height <- 12
dpi <- 600

# functions ---------------------------------------------------------------

load_data <- function(model_type, model_number, ...){
  df1 <- read_csv(here::here(glue::glue("results/{model_number}a_{model_type}_feature_importance.csv")),
                  col_types = cols())
  df2 <- read_csv(here::here(glue::glue("results/{model_number}b_{model_type}_feature_importance.csv")),
                  col_types = cols())
  
  # target <- target[1]
  # if(!(target %in% c("mean", "rank"))){
  #   stop("Oops, wrong target")
  # }
  
  df_pre_long <- df1 %>% 
    pivot_longer(-1) %>% 
    group_by(name) %>% 
    summarise(mean_importance = median(value),
              std = sd(value)/sqrt(n()),
              q1 = quantile(value, 0.25),
              q2 = quantile(value, 0.75),
    ) %>% 
    mutate(time="pre-mandate")
  df_post_long <- df2 %>% 
    pivot_longer(-1) %>% 
    group_by(name) %>% 
    summarise(mean_importance = median(value),
              std = sd(value)/sqrt(n()),
              q1 = quantile(value, 0.25),
              q2 = quantile(value, 0.75),) %>% 
    mutate(time="mandate")
  
  df <- df_pre_long %>% 
    arrange(-mean_importance) %>% 
    bind_rows(df_post_long %>% arrange(-mean_importance)) %>% 
    filter(!str_detect(name, "state")) %>%
    group_by(time) %>% 
    slice(1:10) %>% 
    ungroup() %>% 
    mutate(name=fct_reorder(name, mean_importance)) %>% 
    mutate(time = ifelse(time=="mandate", "During mandates", "Before mandates"))
  
  df$mean_importance[df$time=="Before mandates"] <- -df$mean_importance[df$time=="Before mandates"]
  df$q1[df$time=="Before mandates"] <- -df$q1[df$time=="Before mandates"]
  df$q2[df$time=="Before mandates"] <- -df$q2[df$time=="Before mandates"]
  
  # grouping_list <- c(
  #   "Contact behaviour",
  #   "Demographics",
  #   "Wellbeing",
  #   "Mental health",
  #   "Perception of illness threat",
  #   "State",
  #   "Time",
  #   "Trust in government",
  #   "Comorbidities",
  #   "Self protective behaviours"
  # )
  grouping_list <- c(
    "Self protective behaviours",
    "Demographics",
    "Health, mental health and wellbeing",
    "Perception of illness threat",
    "Time",
    "Trust in government"
  )
  
  df <- df %>% 
    mutate(group_var = case_when(
      str_detect(name, "i2|i9|i11|protective") ~ grouping_list[1],
      str_detect(name, "age|house|employ|gender|state") ~ grouping_list[2],
      str_detect(name, "PHQ|cantril|d1") ~ grouping_list[3],
      str_detect(name, "r1") ~ grouping_list[4],
      str_detect(name, "week") ~ grouping_list[5],
      str_detect(name, "WCR") ~ grouping_list[6],
    )) 
  
  
  
  return(df)
}

labels_clean <- function(labels_original){
  labels <- labels_original %>% 
    str_replace_all("_", " ") %>% 
    str_to_title() 
  labels <- str_replace(labels, " Nomask Scale", "")
  
  labels[str_detect(labels, "State")] <- str_c(labels[str_detect(labels, "State")],
                                               ")")
  labels[str_detect(labels, "State")] <- str_replace(labels[str_detect(labels,
                                                                       "State")],
                                                     "State ",
                                                     "State (")
  
  labels[str_detect(labels, "Gender")] <- str_c(labels[str_detect(labels, "Gender")],
                                                ")")
  labels[str_detect(labels, "Gender")] <- str_replace(labels[str_detect(labels,
                                                                        "Gender")],
                                                      "Gender ",
                                                      "Gender (")
  
  labels[str_detect(labels, "I11 Health")] <- str_c(labels[str_detect(labels, "I11 Health")], 
                                                    " To Isolate")
  labels[str_detect(labels, "I11 Health")] <- str_replace(labels[str_detect(labels, "I11 Health")],
                                                          "I11 Health ",
                                                          "")
  
  labels[str_detect(labels, "Employment")] <- str_c(labels[str_detect(labels, "Employment")],
                                                    ")")
  labels[str_detect(labels, "Employment")] <- str_replace(labels[str_detect(labels,
                                                                            "Employment")],
                                                          "Employment Status ",
                                                          "Employment Status \n(")
  
  labels[str_detect(labels, "I9 Health")] <- str_c(labels[str_detect(labels, "I9 Health")],
                                                   ")")
  labels[str_detect(labels, "I9 Health")] <- str_replace(labels[str_detect(labels,
                                                                           "I9 Health")],
                                                         "I9 Health ",
                                                         "Isolate If Unwell (")
  
  labels[str_detect(labels, "I2 Health")] <- "Non-Household Contacts"
  labels[str_detect(labels, "D1")] <- "Has Commorbidities"
  
  labels[str_detect(labels, "R1 1")] <- "Perceived Severity"
  labels[str_detect(labels, "R1 2")] <- "Perceived Susceptibility"
  
  labels[str_detect(labels, "Phq4")] <- str_c(labels[str_detect(labels, "Phq4")],
                                                   ")")
  labels[str_detect(labels, "Phq4")] <- str_replace(labels[str_detect(labels, "Phq4")],
                                                    "Phq4 1 ",
                                                    "Little interest or pleasure \n(")
  labels[str_detect(labels, "Phq4")] <- str_replace(labels[str_detect(labels, "Phq4")],
                                                    "Phq4 2 ",
                                                    "Feeling down or depressed \n(")
  labels[str_detect(labels, "Phq4")] <- str_replace(labels[str_detect(labels, "Phq4")],
                                                    "Phq4 3 ",
                                                    "Feeling nervous or anxious \n(")
  labels[str_detect(labels, "Phq4")] <- str_replace(labels[str_detect(labels, "Phq4")],
                                                    "Phq4 4 ",
                                                    "Worrying (")
  
  
  labels[str_detect(labels, "Wcrex2")] <- str_c(labels[str_detect(labels, "Wcrex2")],
                                                   ")")
  labels[str_detect(labels, "Wcrex2")] <- str_replace(labels[str_detect(labels, "Wcrex2")],
                                                    "Wcrex2 ",
                                                    "Confidence in response\n(")
  
                                                
  
  return(labels)
}

get_tmp_data <- function(df, tmp_importance){
  df_tmp <- df %>% 
    group_by(time) %>% 
    slice(1) %>% 
    mutate(mean_importance = tmp_importance)
  df_tmp$mean_importance[df_tmp$time=="Before mandates"] <- -df_tmp$mean_importance[df_tmp$time=="Before mandates"]
  
  return(df_tmp)
}

get_colour_palette <- function(name="Spectral", method="brewer"){
  groupings <- c(
    "Self protective behaviours",
    "Demographics",
    "Health, mental health and wellbeing",
    "Perception of illness threat",
    "Time",
    "Trust in government"
  )
  
  if(method=="brewer"){
    cols <- RColorBrewer::brewer.pal(length(groupings), name=name) 
  }
  if(method=="hp"){
    cols <- harrypotter::hp(length(groupings), option=name) 
  }
  names(cols) <- groupings
  return(cols)
}

create_tornado_plot <- function(model_type, model_number){
  
  df <- load_data(model_type = model_type, model_number = model_number, target = "mean")
  
  df_tmp <- get_tmp_data(df, round(max(abs(df$q2), abs(df$q1)) + 0.005, 2))
  
  # Make labels pretty
  labels_original <- df %>% 
    pull(name) %>% 
    levels()
  labels <- labels_clean(labels_original)
  
  df <- df %>% 
    mutate(group_var = factor(group_var, levels = names(colour_palette)))
  p <- df %>%
    ggplot(aes(x=mean_importance, y = name, fill = group_var)) +
    geom_col(show.legend = TRUE) +
    geom_point(data=df_tmp, colour=NA) +
    geom_errorbar(aes(xmin = q1, xmax = q2), width = 0.5)+
    facet_wrap(~time, scale="free_x") +
    scale_x_continuous(expand = c(0, 0), 
                       labels = function(x) signif(abs(x), 3)) +
    scale_y_discrete(labels = labels) +
    scale_fill_manual(values = colour_palette) +
    guides(fill = guide_legend(nrow = 2)) +
    # ggtitle(glue::glue("{model_type} - {model_number}")) +
    labs(y = NULL, x = "Mean feature importance", fill = NULL) +
    theme_bw() +
    theme(panel.spacing.x = unit(0, "mm"),
          legend.position = "none",
          text = element_text(size = text_size),
          axis.text.y = element_text(size=10),
          strip.text = element_text(size=22),
          legend.text = element_text(size=14),
          plot.margin = margin(t=1, r=15, b=1, l=1))
  
  return(p)
  
  # ggsave(here::here(glue::glue("figures/mean_feature_importance_{model_number}_{model_type}.png")),
  #        height = height,
  #        width = 1.618 * height,
  #        dpi = 300)
}

colour_palette <- get_colour_palette(name="Ravenclaw", method="hp")

# xgboost, model 1 --------------------------------------------------------

model_number <- "model_1"
model_type <- "xgboost"

p_xgb <- create_tornado_plot(model_number = model_number, model_type = model_type) +
  theme(legend.position = "bottom")
# rf, model 1 --------------------------------------------------------

model_type <- "rf"

p_rf <- create_tornado_plot(model_number = model_number, model_type = model_type) 

p <- (p_rf/p_xgb) +  
  plot_annotation(tag_levels = 'a', tag_suffix = ")") 


ggsave(here::here(glue::glue("figures/mean_feature_importance_{model_number}_both_models.png")),
       plot = p,
       height = height,
       width = height,
       dpi = dpi)

# xgboost, model 2 --------------------------------------------------------

model_number <- "model_2"
model_type <- "xgboost"

p_xgb <- create_tornado_plot(model_number = model_number, model_type = model_type) +
  theme(legend.position = "bottom")


# rf, model 2 --------------------------------------------------------

model_type <- "rf"

p_rf <- create_tornado_plot(model_number = model_number, model_type = model_type) 

p <- (p_rf/p_xgb) +  
  plot_annotation(tag_levels = 'a', tag_suffix = ")") 

ggsave(here::here(glue::glue("figures/mean_feature_importance_{model_number}_both_models.png")),
       plot = p,
       height = height,
       width = height,
       dpi = dpi)
