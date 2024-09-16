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

# Data --------------------------------------------------------------------

df_mandates <- read_csv(here::here("raw_data/OxCGRT_AUS_latest.csv"),
                        col_types = cols())

df_mandates <- df_mandates %>% 
  select("RegionName", "RegionCode", "Date", "H6M_Facial Coverings") %>% 
  janitor::clean_names() %>% 
  mutate(date = lubridate::ymd(date)) %>% 
  rename(state=region_name) %>% 
  group_by(state) %>% 
  mutate(rolling_avg_mandate = zoo::rollmean(h6m_facial_coverings, 
                                             k=14, 
                                             align="right", 
                                             na.pad=TRUE)) %>% 
  ungroup() %>% 
  drop_na(state) 

period_start_date <- df_mandates %>% 
  group_by(state) %>% 
  mutate(within_period = ifelse(rolling_avg_mandate >= 3, 1, 0)) %>% 
  filter(within_period==1) %>% 
  slice(1) %>% 
  ungroup()

df_mandates <- df_mandates %>% 
  mutate(within_period = map2_dbl(date, state,
                                  function(d, s){
                                    period_start <- period_start_date %>% 
                                      filter(state==s) %>% 
                                      pull(date)
                                    if(d < period_start){
                                      return(0)
                                    }else{
                                      return(1)
                                    }
                                  }),
         within_period = ifelse(within_period==1, "After mandates", "Pre-mandates"))%>% 
  filter(date <= "2022-03-29", date >= "2020-06-24")


# Plot --------------------------------------------------------------------

p <- df_mandates %>% 
  ggplot(aes(x=date, y=h6m_facial_coverings, colour=within_period)) +
  geom_line() +
  geom_line(aes(y=rolling_avg_mandate), colour="grey30", lty=2) +
  facet_wrap(~state, ncol = 2) +
  labs(x=NULL,
       y="Facial covering policy stringency",
       colour = "Period") +
  scale_colour_discrete(breaks = c("Pre-mandates", "After mandates")) +
  theme_bw() +
  theme(legend.position = "bottom",
        text = element_text(size = text_size, family="Times New Roman"))

ggsave(here::here(glue::glue("figures/mandate_periods_time.png")),
       plot = p,
       height = height,
       width = height,
       dpi = dpi)
