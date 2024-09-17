cd "/Users/boyuchen/Documents/UBC/RA/minimum_wage"

local scenarios `1'

* Clear the existing Excel file or create a new one
putexcel set tables/glm_coefficients.xlsx, replace
// putexcel set tables/ols_coefficients.xlsx, replace
// putexcel set tables/lpm_coefficients.xlsx, replace

foreach s in `scenarios'{
quietly{
    
	di "Run GLM for scenario `s'"
	use data/df_grouped_`s'.dta, clear
	drop min_bin min_sum
	
	su wagcat
	local maxbin = `r(max)'
	
	glm fweight min* bin* if remain>0 & wagcat ~= 0 & wagcat ~=  `maxbin', link(cloglog) family(binomial remain) 
	
    * Extract the coefficients
    matrix b = e(b)
	
	* Extract the variance-covariance matrix
    matrix V = e(V)

    * Calculate standard errors (sqrt of diagonal elements of V)
    matrix D = vecdiag(V) // Get the diagonal elements
    matrix se = J(1, colsof(D), .)
    forvalues i = 1/`=colsof(D)' {
        matrix se[1,`i'] = sqrt(D[1,`i'])

    * Export the coefficient matrix to the specified sheet in the Excel file
    putexcel set tables/glm_coefficients.xlsx, sheet(`s') modify
    putexcel A1 = matrix(b), names

    * Export the standard errors below the coefficients
    putexcel B4 = matrix(se)
	putexcel A3 = "coef"
	putexcel A4 = "se"

	* Generate fitted values
	cap drop fitted_fweight
	
	predict fitted_fweight
	
	order fitted_fweight fweight
	
	* Save the dataset with fitted values
	save data/output/df_grouped_fitted_`s'.dta, replace
	}
	}
}
