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
