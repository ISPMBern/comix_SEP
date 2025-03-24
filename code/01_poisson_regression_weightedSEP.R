library(tidyverse)

#for (pop_density in c(FALSE, TRUE)) {
for (pop_density in c(FALSE)) {
  #for (cont_filter in c("cont_overall", "cont_outside_home_exclnan", "cont_outside_home_inclnan", "cont_work")) {
  for (cont_filter in c("cont_overall")) {
    
    data = read_csv(paste0('./../data/processed/comixdata_part_cleaned+sep_munic+', cont_filter, '.csv'))
    
    colnames(data)
    
    list_covariates <- c('panel_wave',  'age_group_part', 
                         'gender_responder', 'region', 'country_cat_birth', 
                         'sep3_quartile', 
                         'education_level3_v3', 'household_income_3cat', 
                         'household_size_cat_v2', 'employment_cat_v2', 
                         'vaccinated', 'weekend')
    
    if (pop_density) {
      list_covariates <- c('panel_wave',  'age_group_part', 
                           'gender_responder', 'region', 'country_cat_birth', 
                           'sep3_quartile', 
                           'education_level3_v3', 'household_income_3cat', 
                           'household_size_cat_v2', 'employment_cat_v2', 
                           'vaccinated', 'weekend', 
                           'pop_densit_cat3')
    }
    
    data$country_cat_birth <- relevel(factor(data$country_cat_birth), ref = "Switzerland")
    
    #data$education_level3<- relevel(factor(data$education_level3), ref = "Obligatory school and vocational education")
    data$education_level3_v3 <- factor(data$education_level3_v3, levels = c("Without tertiary education", 
                                                                            "Tertiary level"))
    
    data$household_income_3cat <- factor(data$household_income_3cat, 
                                         levels = c("0-5,000", "5,001-10,000", "10,000+", "Preferred not to answer"))
    
    data$sep3_quartile <- factor(data$sep3_quartile, 
                                 levels = c("lowest", "highest"))
    
    unique(data$age_group_part)
    
    data$age_group_part <- factor(data$age_group_part, 
                                  levels = c("25-64", "0-14", "15-24", "65+"))
    
    # univariate regression model
    output_x <- c()
    for(i in list_covariates){
      output <- c()
      
      sub_set <- data[c('contacts_fill0_trunc50', list_covariates)]
      colnames(sub_set)[grep(i,colnames(sub_set))] <- "variable"
      
      mod_uni <- glm(formula = contacts_fill0_trunc50 ~ variable,
                     family = poisson(link = "log"),
                     data = sub_set)
      
      output <- data.frame(matrix(0, ncol = 6, nrow = length(names(mod_uni$coefficients))))
      colnames(output) <- c("Variables","names","RR","CI.2.5.", "CI.97.5.", "p.value")
      sum_reg <- summary(mod_uni)
      output$Variables <- i
      output$names <- names(coef(mod_uni))
      output$RR <- coef(mod_uni)
      output$CI.2.5. <-  coef(mod_uni) - qnorm(0.975) * sqrt(diag(vcov(mod_uni)))  #ci(mod_uni)[-c(1),3]
      output$CI.97.5. <- coef(mod_uni) + qnorm(0.975) * sqrt(diag(vcov(mod_uni))) # ci(mod_uni)[-c(1),4]
      output[,c(3:5)] <- exp(output[,c(3:5)]) ### take the exponential of the coefficient, for the RR and CI
      #output$p.value <- sum_reg$coefficients[,"Pr(>|z|)"]
      output$RRCI <- paste0(format(round(output$RR,2), nsmall = 2), " (", format(round(output$CI.2.5.,2), nsmall = 2), "-", format(round(output$CI.97.5.,2), nsmall = 2), ")")
      
      output_x <-rbind(output_x, output)
    }
    
    output_unadj <- output_x
    
    
    # multivariate regression model, with or without interaction term with age
    for (n in 1:2) {
      output_x <- c()
      
      # model without interaction term with age
      if(n==1){
        sub_set <- data[c('contacts_fill0_trunc50', list_covariates)]
        mod_multi <- glm(
          formula = contacts_fill0_trunc50 ~  panel_wave + age_group_part+ sep3_quartile+ 
          gender_responder+ region+ country_cat_birth+ education_level3_v3+
          household_income_3cat+ household_size_cat_v2+ employment_cat_v2+
          vaccinated + weekend,
          family = poisson(link = "log"),
          data = sub_set)
        
        if (pop_density) {
          mod_multi <- update(mod_multi, . ~ . + pop_densit_cat3)
        }
      }
      
      # model with interaction term with age
      # N.B in R formula age*SEP= age + SEP + age:SEP
      if(n==2){
        sub_set <- data[c('contacts_fill0_trunc50', list_covariates)]
        mod_multi <- glm(
          formula = contacts_fill0_trunc50 ~  age_group_part*education_level3_v3+ age_group_part*sep3_quartile+ 
          gender_responder+ region+ country_cat_birth+ panel_wave+
          household_income_3cat+ household_size_cat_v2+ employment_cat_v2+
          vaccinated + weekend, # + pop_densit_cat3,
          family = poisson(link = "log"),
          data = sub_set)
        
        if (pop_density) {
          mod_multi <- update(mod_multi, . ~ . + pop_densit_cat3)
        }
      }
      
      output <- data.frame(matrix(0, ncol = 6, nrow = length(names(mod_multi$coefficients))))
      colnames(output) <- c("Variables","names", "RR","CI.2.5.", "CI.97.5.", "p.value")
      sum_reg <- summary(mod_multi)
      output$names <- names(coef(mod_multi))
      output$RR <- coef(mod_multi)
      output$CI.2.5. <-  coef(mod_multi) - qnorm(0.975) * sqrt(diag(vcov(mod_multi)))  #ci(mod_uni)[-c(1),3]
      output$CI.97.5. <- coef(mod_multi) + qnorm(0.975) * sqrt(diag(vcov(mod_multi))) # ci(mod_uni)[-c(1),4]
      output[,c(3:5)] <- exp(output[,c(3:5)]) ### take the exponential of the coefficient, for the RR and CI
      #output$p.value <- sum_reg$coefficients[,"Pr(>|z|)"]
      output$RRCI <- paste0(format(round(output$RR,2), nsmall = 2), " (", format(round(output$CI.2.5.,2), nsmall = 2), "-", format(round(output$CI.97.5.,2), nsmall = 2), ")")
      output_x <-rbind(output_x, output)
      
      if (n==2) {
        for (g in c('0-14', '15-24', '65+')) {
          #Compute the Standard Error (SE) of the Combined Log Relative Risk:
          #The standard error of the combined log relative risk can be computed using the variance-covariance matrix of the estimated coefficients. 
          #The formula is:
          #se <- sqrt(var(edu) + var(edu*age)) + 2*covar(edu, edu*age))
          #then we can compute the CI as 
          # [ coef(edu) + coef(edu*age) ] +-qnorm * se 
          output <- data.frame(matrix(0, ncol = 6, nrow = 1))
          colnames(output) <- c("Variables","names", "RR","CI.2.5.", "CI.97.5.", "p.value")
          tag1 <- "education_level3_v3Tertiary level"
          tag2 <- paste0("age_group_part", g, ":education_level3_v3Tertiary level")
          nome <- paste0("age_group_part", g, "+education_level3_v3Tertiary level")
          output$names <- nome
          output$RR <- coef(mod_multi)[tag1]+coef(mod_multi)[tag2]
          cov_matrix <- vcov(mod_multi)
          se <- sqrt(cov_matrix[tag1,tag1] + cov_matrix[tag2,tag2] + 2*cov_matrix[tag1,tag2])
          output$CI.2.5. <-  output$RR - qnorm(0.975) * se
          output$CI.97.5. <- output$RR + qnorm(0.975) * se
          output[,c(3:5)] <- exp(output[,c(3:5)])
          output$p.value <- NA
          output$RRCI <- paste0(format(round(output$RR,2), nsmall = 2), " (", format(round(output$CI.2.5.,2), nsmall = 2), "-", format(round(output$CI.97.5.,2), nsmall = 2), ")")
          output_x <-rbind(output_x, output) 
        }
        for (g in c('0-14', '15-24', '65+')) {  
          output <- data.frame(matrix(0, ncol = 6, nrow = 1))
          colnames(output) <- c("Variables","names", "RR","CI.2.5.", "CI.97.5.", "p.value")
          tag1 <- "sep3_quartilehighest"
          tag2 <- paste0("age_group_part", g, ":sep3_quartilehighest")
          nome <- paste0("age_group_part", g, "+sep3_quartilehighest")
          output$names <- nome
          output$RR <- coef(mod_multi)[tag1]+coef(mod_multi)[tag2]
          cov_matrix <- vcov(mod_multi)
          se <- sqrt(cov_matrix[tag1,tag1] + cov_matrix[tag2,tag2] + 2*cov_matrix[tag1,tag2])
          output$CI.2.5. <-  output$RR - qnorm(0.975) * se
          output$CI.97.5. <- output$RR + qnorm(0.975) * se
          output[,c(3:5)] <- exp(output[,c(3:5)])
          output$p.value <- NA
          output$RRCI <- paste0(format(round(output$RR,2), nsmall = 2), " (", format(round(output$CI.2.5.,2), nsmall = 2), "-", format(round(output$CI.97.5.,2), nsmall = 2), ")")
          output_x <-rbind(output_x, output) 
        }
      }
      
      if(n==1){
        output_adj <- output_x
      }
      if(n==2){
        output_adj_interaction <- output_x
      }
      
    }
    
    if (pop_density) {
      add_tag <- "_pop_density"
    }
    else {
      add_tag <- ""
    }
    
    write.csv(output_unadj, paste0("./../output/regression_output/output_unadj_", cont_filter, add_tag, ".csv"), row.names=FALSE) 
    #write.csv(output_adj, paste0("./../output/regression_output/output_adj_", cont_filter, add_tag, ".csv"), row.names=FALSE)
    write.csv(output_adj_interaction, paste0("./../output/regression_output/output_adj_interaction_", cont_filter, add_tag, ".csv"), row.names=FALSE)
  }  
}

