## Matt Ryan
## Create Mandates plot
## 02/09/2024
## Requires: tidyverse, here, glue


# Libraries ---------------------------------------------------------------
pacman::p_load(tidyverse)


# Plot parameters ---------------------------------------------------------


text_size <- 16
height <- 12
dpi <- 600



# data --------------------------------------------------------------------

df_full <- read_csv("raw_data/australia.csv",
                    col_types = cols()) 

face_mask_items <- df_full %>% 
  select("i12_health_1", "i12_health_22", "i12_health_23", "i12_health_25") %>% 
  mutate(across(everything(),
                ~case_when(
                  .x=="Always" ~ 5,
                  .x=="Frequently"~ 4,
                  .x=="Sometimes"~ 3,
                  .x=="Rarely"~ 2,
                  .x=="Not at all" ~ 1,
                  TRUE ~ NA
                )))

face_mask_cts <- face_mask_items %>% 
  rowMeans(na.rm = T)

df_full <- df_full %>% 
  mutate(face_mask_cts = face_mask_cts,
         face_masks = ifelse(face_mask_cts >= 4, "Yes", "No"),
         endtime = map_chr(endtime, ~str_split(.x, " ")[[1]][1]),
         endtime = lubridate::dmy(endtime),
         FN = floor(as.numeric(endtime - lubridate::ymd("2020-04-01"))/14))

df_sub <- read_csv("data/cleaned_data.csv", col_types = cols())
df_sub <- df_sub %>% 
  mutate(FN = floor(as.numeric(endtime - lubridate::ymd("2020-04-01"))/14))


# Mask signals ------------------------------------------------------------

full_mask_signal <- df_full %>% 
  group_by(state, FN) %>% 
  count(face_masks) %>% 
  mutate(p = n/sum(n),
         p_se = sqrt(p*(1-p)/sum(n))) %>% 
  filter(face_masks=="Yes")

sub_mask_signal <- df_sub %>% 
  group_by(state, FN) %>% 
  count(face_mask_behaviour_binary) %>% 
  mutate(p = n/sum(n)) %>% 
  filter(face_mask_behaviour_binary=="Yes")


# Plot --------------------------------------------------------------------

p <- full_mask_signal %>% 
  ggplot(aes(x=FN, y=p)) +
  geom_ribbon(aes(ymin = p-2*p_se, ymax=p+2*p_se), fill="blue", alpha=0.1) +
  geom_line(colour="blue") +
  geom_line(data=sub_mask_signal, colour="red") +
  facet_wrap(~state, ncol=2) +
  labs(y="Proportion wearing face masks",
       x="Survey fortnight") +
  theme_bw() +
  theme(text = element_text(size=text_size, family="Times New Roman"))


ggsave(here::here(glue::glue("figures/mask_signals.png")),
       plot = p,
       height = height,
       width = height,
       dpi = dpi)

