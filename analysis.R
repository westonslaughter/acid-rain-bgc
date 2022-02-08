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

theme_set(ggthemes::theme_few())


#### Helper Function ####
#load authorization file for macrosheds google sheets
## googlesheets4::gs4_auth(path = '../../data_processing/googlesheet_service_accnt.json')
## sm <- suppressMessages

## #read in secrets
## conf <- jsonlite::fromJSON(readLines('src/config.json'),
##                            simplifyDataFrame = FALSE)

## load_config_datasets(from_where = 'remote')

## site_data <- filter(site_data,
##                     as.logical(in_workflow))

## network_domain <- site_data %>%
##     select(network, domain) %>%
##     distinct() %>%
##     arrange(network, domain)

## dom_net <- site_data %>%
##   select(network, domain, site_code)

## Watershed Attribute Groupings
ws_summaries <- read.csv("src/spatial/watershed_summaries.csv")

#### Data read and munging ####

var_interest <- c('NO3_N', 'NO3_NO2_N', 'Ca', 'Cl', 'Mg', 'discharge', 'Na', 'pH', 'SiO2_Si', 'SiO3_Si', 'Si', "SiO2", 'SiO3', 'K', 'Mg')
all_chem <- tibble()

## for(i in 1:nrow(network_domain)){

##     network <- network_domain[[i,1]]
##     domain <- network_domain[[i,2]]

##     stream_gauges <- site_data %>%
##         filter(site_type == 'stream_gauge',
##                network == !!network,
##                domain == !!domain,
##                in_workflow == 1) %>%
##         pull(site_code)

##     derive_files <- list.files(glue('~/files/projects/science/macrosheds/nitrate/nitrate_patterns/src/macrosheds_dataset_v1/{n}/{d}/derived',
##                                     n = network,
##                                     d = domain),
##                                full.names = TRUE)

##     stream_chemistry <- grep('stream_chemistry', derive_files, value = TRUE)

##     if(!length(stream_chemistry) == 1){
##         stop(glue('{n} {d} is has more than one or is missing a stream chemistry file',
##                   n = network, d = domain))
##     }

##     all_chem_fils <- list.files(stream_chemistry, full.names = TRUE)

##     site_chem <- map_dfr(all_chem_fils, read_feather) %>%
##         # Filter for stream Guages
##         filter(site_code %in% !!stream_gauges)

##     site_chem <- site_chem %>%
##         mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
##         filter(var == !!var_interest)

##     ## if((!'NO3_N' %in% macrosheds::ms_drop_var_prefix(unique(site_chem$var))) &&
##     ##    'NO3_NO2_N' %in% macrosheds::ms_drop_var_prefix(unique(site_chem$var))){

##     ##     site_chem <- site_chem %>%
##     ##         mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
##     ##         filter(var == 'NO3_NO2_N')
##     ## } else{
##     ##     site_chem <- site_chem %>%
##     ##         mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
##     ##         filter(var == !!var_interest)
##     ## }

##     all_chem <- rbind(all_chem, site_chem)
## }

## all_fils <- list.files("src/macrosheds_dataset_v1/lter/hbef/derived/stream_chemistry__ms006/", full.names = TRUE)
## d <- map_dfr(all_fils, read_feather)%>%
##     filter(ms_status == 0,
##            ms_interp == 0) %>%
##     select(-ms_status, -ms_interp, -val_err)  %>%
##     pivot_wider(names_from = 'var', values_from = 'val')

## n <- all_chem %>%
##     mutate(year = year(datetime),
##            month = month(datetime),
##            val = as.numeric(val)) %>%
##     #dplyr::filter(val > 0) %>%
##     dplyr::filter(ms_interp == 0,
##                   ms_status == 0) %>%
##     dplyr::filter(domain != 'neon') %>%
##     select(-val_err)

## var_intrest <- 'discharge'
## all_q <- tibble()

## for(i in 1:nrow(network_domain)){

##     network <- network_domain[[i,1]]
##     domain <- network_domain[[i,2]]

##     stream_gauges <- site_data %>%
##         filter(site_type == 'stream_gauge',
##                network == !!network,
##                domain == !!domain,
##                in_workflow == 1) %>%
##         pull(site_code)

##     derive_files <- list.files(glue('~/files/projects/science/macrosheds/nitrate/nitrate_patterns/src/macrosheds_dataset_v1/{n}/{d}/derived',
##                                     n = network,
##                                     d = domain),
##                                full.names = TRUE)

##     discharge <- grep('discharge', derive_files, value = TRUE)

##     if(!length(discharge) == 1){
##         stop(glue('{n} {d} is has more than one or is missing a stream chemistry file',
##                   n = network, d = domain))
##     }

##     all_q_fils <- list.files(discharge, full.names = TRUE)

##     site_q <- map_dfr(all_q_fils, read_feather) %>%
##         # Filter for stream Guages
##         filter(site_code %in% !!stream_gauges) %>%
##         # Filter varible
##         mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
##         filter(var == !!var_intrest)

##     if('year' %in% names(site_q)){
##         site_q <- site_q %>%
##             select(-year)
##     }

##     all_q <- rbind(all_q, site_q)
## }

## all_q <- all_q %>%
##     mutate(year = year(datetime),
##            month = month(datetime),
##            val = as.numeric(val)) %>%
##     select(-var, q = val, -ms_status, -ms_interp, -val_err)

## qc <- inner_join(n, all_q, by = c()) ## %>%
##     ## filter(!is.na(val))

## write.csv(qc, 'chem.csv')

## hubbard <- qc %>% filter(grepl("^w[0-9]", site_code))
## write.csv(hubbard, 'hubbard_chem.csv')

## casi <- hubbard %>% filter(grepl("Ca|Si", var))
## wide_casi <- casi %>% pivot_wider(names_from = var, values_from = val)

## write.csv(wide_casi, 'casi.csv')


## Holy Grail
all_fils <- list.files("src/macrosheds_dataset_v1/lter/hbef/derived/stream_chemistry__ms006/", full.names = TRUE)

d <- map_dfr(all_fils, read_feather)%>%
    filter(ms_status == 0,
           ms_interp == 0) %>%
    select(-ms_status, -ms_interp, -val_err)  %>%
        mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
        ## filter(var == !!var_intrest) %>%
  pivot_wider(names_from = 'var', values_from = 'val')

hubbard <- d[c("datetime", "site_code", "Ca", "SiO2_Si", "M", "Cl", "")]
