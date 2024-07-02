cd "/Users/boyuchen/Documents/UBC/RA/minimum_wage"

* Clear the existing Excel file or create a new one
putexcel set tables/coefficients.xlsx, replace

foreach s in S1 S2 S3 S4{
    
	di "Run GLM for scenario `s'"
	use data/df_grouped_`s'.dta, clear

    glm fweight min min610b min35b min12b min12a min35a min610a min1115a min1620a bin* if remain>0 & wagcat~=211 & wagcat ~= 0, link(cloglog) family(binomial remain) 

    * Extract the coefficients
    matrix b = e(b)
	
	* Extract the variance-covariance matrix
    matrix V = e(V)

    * Calculate standard errors (sqrt of diagonal elements of V)
    matrix D = vecdiag(V) // Get the diagonal elements
    matrix se = J(1, colsof(D), .)
    forvalues i = 1/`=colsof(D)' {
        matrix se[1,`i'] = sqrt(D[1,`i'])
    }

    * Export the coefficient matrix to the specified sheet in the Excel file
    putexcel set tables/coefficients.xlsx, sheet(`s') modify
    putexcel A1 = matrix(b), names

    * Export the standard errors below the coefficients
    putexcel B4 = matrix(se)
	putexcel A3 = "coef"
	putexcel A4 = "se"
	
}
