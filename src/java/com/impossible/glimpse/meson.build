
conf_data = configuration_data()
if tango_enabled
    conf_data.set('use_tango', 'true')
else
    conf_data.set('use_tango', 'false')
endif
glimpse_config_file = configure_file(input : 'GlimpseConfig.java.in',
                                     output : 'GlimpseConfig.java',
                                     configuration : conf_data)
