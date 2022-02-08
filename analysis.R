library(errors)
library(tidyverse)
library(dygraphs)
library(tsibble)
library(lubridate)
library(plotly)
library(feather)
library(glue)
library(xts)
library(dygraphs)
library(here)
library(stringr)

source("project_helpers.R")

## Watershed Attribute Groupings
ws_summaries <- read.csv("src/spatial/watershed_summaries.csv")

#### Data read and munging ####
var_interest <- c('NO3_N', 'NO3_NO2_N', 'Ca', 'Cl', 'Mg', 'discharge', 'Na', 'pH', 'SiO2_Si', 'SiO3_Si', 'Si', "SiO2", 'SiO3', 'K', 'Mg')
chem_fils <- list.files("src/macrosheds_dataset_v1/lter/hbef/derived/stream_chemistry__ms006/", full.names = TRUE)
q_fils <- list.files("src/macrosheds_dataset_v1/lter/hbef/derived/discharge__ms003/", full.names = TRUE)

hbef_chem <- map_dfr(chem_fils, read_feather)%>%
    filter(ms_status == 0,
           ms_interp == 0) %>%
    select(-ms_status, -ms_interp, -val_err)  %>%
    mutate(var = macrosheds::ms_drop_var_prefix(var),
         year = year(datetime),
         month = month(datetime),
         val = as.numeric(val)) %>%
        ## filter(var == !!var_intrest) %>%
  pivot_wider(names_from = 'var', values_from = 'val')


hbef_q <- map_dfr(q_fils, read_feather) %>%
  mutate(var = macrosheds::ms_drop_var_prefix(var),
         year = year(datetime),
         month = month(datetime),
         val = as.numeric(val)) %>%
  filter(ms_status == 0,
         ms_interp == 0) %>%
  select(-ms_status, -ms_interp, -val_err) %>%
  pivot_wider(names_from = 'var', values_from = 'val')

hbef <- merge(hbef_chem, hbef_q, by = c("datetime", "year", "month", "site_code"))
hbef_elements <- hbef[c("datetime", "year", "month", "site_code", "discharge",
                     "Ca", "SiO2_Si", "Mg", "Cl", "Na", "K")]

write.csv(hbef, "hbef.csv")
write.csv(hbef_elements, "hbef_elements.csv")
