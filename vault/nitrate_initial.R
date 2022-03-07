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

theme_set(ggthemes::theme_few())


#### Helper Function ####
#load authorization file for macrosheds google sheets
googlesheets4::gs4_auth(path = '../../data_processing/googlesheet_service_accnt.json')

sm <- suppressMessages

load_config_datasets <- function(from_where){
    
    #this loads our "configuration" datasets into the global environment.
    #as of 10/27/20 those datasets include site_data, variables, universal_products,
    #name_variants (which is not loaded), and
    #watershed_delineation_specs. depending on the type of instance (remote/local),
    #those datasets are either read as local CSVs or as google sheets. for ms
    #developers, this will always be "remote". for future users, it'll be a
    #configurable option.
    
    if(from_where == 'remote'){
        
        ms_vars <- sm(googlesheets4::read_sheet(
            conf$variables_gsheet,
            na = c('', 'NA'),
            col_types = 'cccccccnnccnn'
        ))
        
        site_data <- sm(googlesheets4::read_sheet(
            conf$site_data_gsheet,
            na = c('', 'NA'),
            col_types = 'ccccccccnnnnnccc'
        ))
        
        ws_delin_specs <- sm(googlesheets4::read_sheet(
            conf$delineation_gsheet,
            na = c('', 'NA'),
            col_types = 'cccncnnccl'
        ))
        
        univ_products <- sm(googlesheets4::read_sheet(conf$univ_prods_gsheet,
                                                      na = c('', 'NA')))
        
    } else if(from_where == 'local'){
        
        ms_vars <- sm(read_csv('data/general/variables.csv'))
        site_data <- sm(read_csv('data/general/site_data.csv'))
        univ_products <- sm(read_csv('data/general/universal_products.csv'))
        ## univ_products <- sm(read_csv('data/general/universal_products.csv'))
        
        ws_delin_specs <- tryCatch(sm(read_csv('data/general/watershed_delineation_specs.csv')),
                                   error = function(e){
                                       empty_tibble <- tibble(network = 'a',
                                                              domain = 'a',
                                                              site_code = 'a',
                                                              buffer_radius_m = 1,
                                                              snap_method = 'a',
                                                              snap_distance_m = 1,
                                                              dem_resolution = 1,
                                                              flat_increment = 1,
                                                              breach_method = 'a',
                                                              burn_streams = 'a')
                                       
                                       return(empty_tibble[-1, ])
                                   })
        
    } else {
        stop('from_where must be either "local" or "remote"')
    }
    
    assign('ms_vars',
           ms_vars,
           pos = .GlobalEnv)
    
    assign('site_data',
           site_data,
           pos = .GlobalEnv)
    
    assign('ws_delin_specs',
           ws_delin_specs,
           pos = .GlobalEnv)
    
    assign('univ_products',
           univ_products,
           pos = .GlobalEnv)
}

#read in secrets
conf <- jsonlite::fromJSON(readLines('src/config.json'),
                           simplifyDataFrame = FALSE)

load_config_datasets(from_where = 'remote')

site_data <- filter(site_data,
                    as.logical(in_workflow))

network_domain <- site_data %>%
    select(network, domain) %>%
    distinct() %>%
    arrange(network, domain)

dom_net <- site_data %>%
    select(network, domain, site_code)

plot_chem <- function(site_code, years = NULL){
    
    site <- all_chem %>%
        filter(site_code == !!site_code) %>%
        filter(!is.na(val))
    
    if(!is.null(years)){
        site <- site %>%
            filter(lubridate::year(datetime) %in% !!years)
    }
    fin_plot <- xts(x = site$val, order.by = site$datetime)
    
    dygraph(fin_plot) 
    
}


#### Data read and munging ####

var_intrest <- 'NO3_N'
all_chem <- tibble()

for(i in 1:nrow(network_domain)){
    
    network <- network_domain[[i,1]]
    domain <- network_domain[[i,2]]
    
    stream_gauges <- site_data %>%
        filter(site_type == 'stream_gauge',
               network == !!network,
               domain == !!domain,
               in_workflow == 1) %>%
        pull(site_code)
    
    derive_files <- list.files(glue('~/files/projects/science/macrosheds/nitrate/nitrate_patterns/src/macrosheds_dataset_v1/{n}/{d}/derived',
                                    n = network,
                                    d = domain),
                               full.names = TRUE)
    
    stream_chemistry <- grep('stream_chemistry', derive_files, value = TRUE)
    
    if(!length(stream_chemistry) == 1){
        stop(glue('{n} {d} is has more than one or is missing a stream chemistry file',
                  n = network, d = domain))
    }
    
    all_chem_fils <- list.files(stream_chemistry, full.names = TRUE)
    
    site_chem <- map_dfr(all_chem_fils, read_feather) %>%
        # Filter for stream Guages
        filter(site_code %in% !!stream_gauges) 
    
    if((!'NO3_N' %in% macrosheds::ms_drop_var_prefix(unique(site_chem$var))) && 
       'NO3_NO2_N' %in% macrosheds::ms_drop_var_prefix(unique(site_chem$var))){
        
        site_chem <- site_chem %>%
            mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
            filter(var == 'NO3_NO2_N')
    } else{
        site_chem <- site_chem %>%
            mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
            filter(var == !!var_intrest)
    }
    
    all_chem <- rbind(all_chem, site_chem)
}



n <- all_chem %>%
    mutate(year = year(datetime),
           month = month(datetime),
           val = as.numeric(val)) %>%
    #dplyr::filter(val > 0) %>%
    dplyr::filter(ms_interp == 0,
                  ms_status == 0) %>%
    dplyr::filter(domain != 'neon') %>%
    select(-val_err)

var_intrest <- 'discharge'
all_q <- tibble()
for(i in 1:nrow(network_domain)){
    
    network <- network_domain[[i,1]]
    domain <- network_domain[[i,2]]
    
    stream_gauges <- site_data %>%
        filter(site_type == 'stream_gauge',
               network == !!network,
               domain == !!domain,
               in_workflow == 1) %>%
        pull(site_code)
    
    derive_files <- list.files(glue('~/files/projects/science/macrosheds/nitrate/nitrate_patterns/src/macrosheds_dataset_v1/{n}/{d}/derived',
                                    n = network,
                                    d = domain),
                               full.names = TRUE)
    
    discharge <- grep('discharge', derive_files, value = TRUE)
    
    if(!length(discharge) == 1){
        stop(glue('{n} {d} is has more than one or is missing a stream chemistry file',
                  n = network, d = domain))
    }
    
    all_q_fils <- list.files(discharge, full.names = TRUE)
    
    site_q <- map_dfr(all_q_fils, read_feather) %>%
        # Filter for stream Guages
        filter(site_code %in% !!stream_gauges) %>%
        # Filter varible 
        mutate(var = macrosheds::ms_drop_var_prefix(var)) %>%
        filter(var == !!var_intrest)
    
    if('year' %in% names(site_q)){
        site_q <- site_q %>%
            select(-year)
    }
    
    all_q <- rbind(all_q, site_q)
}

all_q <- all_q %>%
    mutate(year = year(datetime),
           month = month(datetime),
           val = as.numeric(val)) %>%
    select(-var, q = val, -ms_status, -ms_interp, -val_err)


qc <- inner_join(n,all_q, by = c()) %>%
    filter(!is.na(val))



#### Data exploration ####

## QC Plots
qc %>%
    # dplyr::filter(domain == 'east_river') %>%
    #dplyr::filter(year > 2010)  %>%
    ggplot(aes(x = q, y = val, color = month)) +
    geom_point(shape = 1, size = 1, alpha = 0.5) +
    facet_wrap(~site_code, scales = 'free') +
    scale_x_log10() +
    scale_y_log10() +
    scale_color_gradient2(low = 'red3',mid = 'gray40',high = 'blue', midpoint = 6)


## Monthly summaries

month_n <- n %>%
    left_join(dom_net) %>%
    # dplyr::filter(year >= 2010) %>%
    group_by(network, domain, site_code) %>%
    mutate(decade_mean = mean(val,na.rm=T)) %>%
    group_by(network, domain, site_code, month) %>%
    mutate(norm_val = val/decade_mean,
           count = n()) %>%
    dplyr::filter(count > 5) %>%
    group_by(site_code) %>%
    mutate(unique_mo = n_distinct(month)) %>%
    dplyr::filter(unique_mo > 8) %>%
    group_by(network, domain, site_code, month) %>%
    summarize(mean_n = mean(norm_val, na.rm = T),
              max_n = max(norm_val, na.rm = T),
              min_n = min(norm_val, na.rm = T),
              median_n = median(norm_val, na.rm = T)) %>%
    group_by(site_code) %>%
    mutate(season = 
               case_when(month %in% c(12,1,2) ~ 'winter',
                         month %in% c(3,4,5) ~ 'spring',
                         month %in% c(6,7,8) ~ 'summer',
                         month %in% c(9,10,11) ~ 'fall')) %>%
    mutate(max_mean = max(mean_n)) 



max_means_only <- month_n %>%
    dplyr::filter(max_mean == mean_n) %>%
    mutate(n_peak = paste(season,'peak')) %>%
    select(site_code,n_peak)


month_peaks <- left_join(month_n,max_means_only)

### Pattern check

g1 <- ggplot(month_peaks %>%
                 dplyr::filter(!is.na(n_peak)), aes(x=month,y = mean_n,
                                                    group = site_code,
                                                    color = domain)) + 
    geom_line() + 
    facet_wrap(~n_peak,ncol= 2, scales = 'free') + 
    ylab('Monthly average nitarate (> 2010)')

g1


table(max_means_only$n_peak)

plotly::ggplotly(g1)

View(max_means_only)



plot_chem('Q1') 



## Next ideas

## Seasonal Q vs Seasonal N conc (agree or disagree)

## Temp synchrony or not.

## Watershed Attribute Groupings
ws_summaries <- read.csv("src/spatial/watershed_summaries.csv")

# Mean Monthly Nitrate by Site and Domain
# Watershed Attribute Quartile Groupings
## attribute loop
ws_attrs <- c("ws_area_ha", "cc_mean_annual_precip", "cc_mean_annual_temp", "va_mean_annual_gpp")

for (ws_var in ws_attrs) {
  name <- paste0(ws_var, "_quartile")
  quartile <-  "quartile"
  ## assign(name, ws_summaries[ws_var])
  if(is.numeric(ws_summaries[[ws_var]])) {
    ws_summaries["quartile"] <- cut(ws_summaries[[ws_var]],
                             quantile(ws_summaries[ws_var], na.rm=TRUE),
                             include.lowest = TRUE,
                             labels = FALSE,
                             na.rm=TRUE)

    ws_this <- ws_summaries[, c("network", "domain", "site_code", quartile)] %>%
      merge(month_peaks, by=c("network", "domain", "site_code"))

    axis <- paste("monthly avg nitrate (after 2010) \n")
    quartile.labs <- c("Lowest Quartile", "Second Quartile", "Third Quartile", "Highest Quartile")
    names(quartile.labs) <- c(1, 2, 3, 4)


    ws_facet <- ggplot(ws_this %>%
                 dplyr::filter(!is.na(quartile)), aes(x=month, y = mean_n,
                                                    group = site_code,
                                                    color = domain)) +
    geom_line() +
    facet_wrap(~quartile, ncol= 2, scales = 'free', labeller = labeller(quartile = quartile.labs)) +
      ylab(axis) +
      scale_x_discrete(limits = month.abb) +
      ggtitle(name) +
      theme(
        plot.margin = unit(c(2,2,2,2),"cm"),
        plot.title = element_text(size = 22, face = "bold", margin = margin(t = 0.5, b = 0.5, unit = "cm")),
        axis.title = element_text(margin = margin(t = 0.5, b = 0.5, unit = "cm"), size = 20),
        panel.spacing = unit(0.75, "lines"),
        axis.title.x = element_text(margin=margin(t=1)), #add margin to x-axis
        axis.title.y = element_text(margin=margin(r=1)), #add margin to y-axis title
        strip.text.x = element_text(
        size = 18, color = "black", face = "bold.italic"
        ),
      strip.text.y = element_text(
        size = 18, color = "black", face = "bold.italic"
      ),
      axis.text.x = element_text(face="bold", color="#000000",
                           size=14, angle=45, margin = margin(t = 0.5, b = 0.5, unit = "cm"), hjust = 0.5),
      axis.text.y = element_text(face="bold", color="#000000",
                           size=14, angle=45, margin = margin(t = 0.5, b = 0.5, unit = "cm"), hjust = 0.5)
      
      )

    plotfile <- paste0(name, ".png")
    ## ggsave(file = filename, dpi = 600, width = 15, height = 20, units = "in")
    ggsave(filename = here("figures/intro_plots/",plotfile), width = 3840/276, height = 2400/276, dpi = 276)
    ## plotly::ggplotly(ws_area_facet)
  } else {
    statement <- paste(ws_var, "is not numeric, column skipped")
    print(statement)
  }
}

# Mean Monthly Nitrate by Site and Domain
# Watershed Attribute Coloring

ws_nitrate_line <- function(df, summary, attribute) {

      summary["quartile"] <- cut(summary[[attribute]],
                             quantile(summary[attribute], na.rm=TRUE),
                             include.lowest = TRUE,
                             labels = FALSE,
                             na.rm=TRUE)

    ws_this <- summary[, c("network", "domain", "site_code", "quartile")] %>%
      merge(df, by=c("network", "domain", "site_code"))

    axis <- paste("monthly avg nitrate (after 2010) \n")
    quartile.labs <- c("Lowest Quartile", "Second Quartile", "Third Quartile", "Highest Quartile")
    names(quartile.labs) <- c(1, 2, 3, 4)

    ws_line <- ggplot(ws_this %>%
                 dplyr::filter(!is.na(quartile)), aes(x=month, y = mean_n,
                                                    group = site_code,
                                                    color = as.character(quartile))) +
      geom_line() +
      ylab(axis) +
      scale_x_discrete(limits = month.abb)


   return(ws_line)
}


# Mean Monthly Nitrate by Site and Domain FUNCTION
# Watershed Attribute Quartile Groupings
## attribute loop
ws_attrs <- c("ws_area_ha", "cc_mean_annual_precip", "cc_mean_annual_temp", "va_mean_annual_gpp")

ws_quartiler <- function(df, summary, facet_attr, col_attr) {
  ## Q1: facet
  name <- paste0(facet_attr, "_quartile")
  quartile <-  "quartile"
  ## assign(name, summary[facet_attr])
  if(is.numeric(summary[[facet_attr]])) {
    summary["quartile"] <- cut(summary[[facet_attr]],
                             quantile(summary[facet_attr], na.rm=TRUE),
                             include.lowest = TRUE,
                             labels = FALSE,
                             na.rm=TRUE)

    ws_this <- summary[, c("network", "domain", "site_code", quartile)] %>%
      merge(df, by=c("network", "domain", "site_code"))

    axis <- paste("monthly avg nitrate (after 2010) \n")
    quartile.labs <- c("Lowest Quartile", "Second Quartile", "Third Quartile", "Highest Quartile")
    names(quartile.labs) <- c(1, 2, 3, 4)

    ## Q2: color
    name <- paste0(col_attr, "_quartile_col")
    quartile_col <-  "quartile_col"

    ## assign(name, summary[col_attr])
    summary["quartile_col"] <- cut(summary[[col_attr]],
                             quantile(summary[col_attr], na.rm=TRUE),
                             include.lowest = TRUE,
                             labels = FALSE,
                             na.rm=TRUE)

    summary <- summary %>%
      mutate(quartile_col = as.character(quartile_col))

    ws_this <- summary[, c("network", "domain", "site_code", quartile, quartile_col)] %>%
      merge(df, by=c("network", "domain", "site_code"))

    quartile_col.labs <- c("Lowest Quartile_Col", "Second Quartile_Col", "Third Quartile_Col", "Highest Quartile_Col")
    names(quartile_col.labs) <- c(1, 2, 3, 4)

    ws_facet <- ggplot(ws_this %>%
                 dplyr::filter(!is.na(quartile)), aes(x=month, y = mean_n,
                                                    group = site_code,
                                                    color = n_peak)) +
    geom_line() +
    facet_wrap(~quartile, ncol= 2, scales = 'free', labeller = labeller(quartile = quartile.labs)) +
      ylab(axis) +
      scale_x_discrete(limits = month.abb) +
      ggtitle(name) +
      theme(
        plot.margin = unit(c(2,2,2,2),"cm"),
        plot.title = element_text(size = 22, face = "bold", margin = margin(t = 0.5, b = 0.5, unit = "cm")),
        axis.title = element_text(margin = margin(t = 0.5, b = 0.5, unit = "cm"), size = 20),
        panel.spacing = unit(0.75, "lines"),
        axis.title.x = element_text(margin=margin(t=1)), #add margin to x-axis
        axis.title.y = element_text(margin=margin(r=1)), #add margin to y-axis title
        strip.text.x = element_text(
        size = 18, color = "black", face = "bold.italic"
        ),
      strip.text.y = element_text(
        size = 18, color = "black", face = "bold.italic"
      ),
      axis.text.x = element_text(face="bold", color="#000000",
                           size=14, angle=45, margin = margin(t = 0.5, b = 0.5, unit = "cm"), hjust = 0.5),
      axis.text.y = element_text(face="bold", color="#000000",
                           size=14, angle=45, margin = margin(t = 0.5, b = 0.5, unit = "cm"), hjust = 0.5)

      )

      return(ws_facet)

  } else {
    statement <- paste(facet_attr, "is not numeric, column skipped")
    print(statement)
  }

}


  name <- paste0(ws_var, "_quartile")
  quartile <-  "quartile"
  ## assign(name, ws_summaries[ws_var])
  ws_summaries["quartile"] <- cut(ws_summaries[["va_mean_annual_gpp"]],
                             quantile(ws_summaries["va_mean_annual_gpp"], na.rm=TRUE),
                             include.lowest = TRUE,
                             labels = FALSE,
                             na.rm=TRUE)

    ws_this <- ws_summaries[, c("network", "domain", "site_code", "va_mean_annual_gpp", quartile)] %>%
      merge(month_peaks, by=c("network", "domain", "site_code"))

    ws_this <- ws_this %>% dplyr::filter(!is.na(quartile))

    axis <- paste("monthly avg nitrate (after 2010) \n")
    quartile.labs <- c("Lowest Quartile", "Second Quartile", "Third Quartile", "Highest Quartile")
    names(quartile.labs) <- c(1, 2, 3, 4)


    ws_facet <- ggplot(ws_this %>%
                 dplyr::filter(!is.na(n_peak)), aes(x=month, y = mean_n,
                                                    group = site_code,
                                                    color = as.factor(n_peak))) +
    geom_line() +
    facet_wrap(~quartile, ncol= 2, scales = 'free', labeller = labeller(quartile = quartile.labs)) +
      ylab(axis) +
      scale_x_discrete(limits = month.abb) +
      scale_color_manual(values=c('#f2a21a', '#61f428', '#FFD700', '#abd6ff'))+
      ggtitle(name) +
      theme(
        plot.margin = unit(c(2,2,2,2),"cm"),
        plot.title = element_text(size = 22, face = "bold", margin = margin(t = 0.5, b = 0.5, unit = "cm")),
        axis.title = element_text(margin = margin(t = 0.5, b = 0.5, unit = "cm"), size = 20),
        panel.spacing = unit(0.75, "lines"),
        axis.title.x = element_text(margin=margin(t=1)), #add margin to x-axis
        axis.title.y = element_text(margin=margin(r=1)), #add margin to y-axis title
        strip.text.x = element_text(
        size = 18, color = "black", face = "bold.italic"
        ),
      strip.text.y = element_text(
        size = 18, color = "black", face = "bold.italic"
      ),
      axis.text.x = element_text(face="bold", color="#000000",
                           size=14, angle=45, margin = margin(t = 0.5, b = 0.5, unit = "cm"), hjust = 0.5),
      axis.text.y = element_text(face="bold", color="#000000",
                                 size=14, angle=45, margin = margin(t = 0.5, b = 0.5, unit = "cm"), hjust = 0.5),
      legend.key.size = unit(4, 'cm')

      )

    plotfile <- paste0("nitrate_peakseason_gpp", ".png")
    ## ggsave(file = filename, dpi = 600, width = 15, height = 20, units = "in")
    ggsave(filename = here("figures/intro_plots/",plotfile), width = 3840/276, height = 2400/276, dpi = 276)
    ## plotly::ggplotly(ws_area_facet)
