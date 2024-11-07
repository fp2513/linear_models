20241107_linear_models
================
2024-11-07

In airbnb dataset, see how different rental prices differ

Doing some data cleaning first

``` r
data("nyc_airbnb")

nyc_airbnb = 
  nyc_airbnb %>%  
  mutate(stars = review_scores_location / 2) %>%  
  rename(
    borough = neighbourhood_group,
    neighborhood = neighbourhood) %>%  
  filter(borough != "Staten Island") %>%  
  select(price, stars, borough, neighborhood, room_type) %>% 
  mutate(
    borough = fct_infreq(borough),
    room_type = fct_infreq(room_type)
  )
```

## Fit some models

consider price as an outcome that may depend on rating We fit that
initial model in the following code.

``` r
fit = lm(price ~ stars, data = nyc_airbnb)
```

Giving us: Coefficients: (Intercept) stars  
-66.50 44.11

other things that we can ask

``` r
fit = lm(price ~ stars, data = nyc_airbnb)

summary(fit)
```

    ## 
    ## Call:
    ## lm(formula = price ~ stars, data = nyc_airbnb)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -144.1  -69.1  -32.0   25.9 9889.0 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -66.500     11.893  -5.591 2.27e-08 ***
    ## stars         44.115      2.515  17.538  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 183.8 on 30528 degrees of freedom
    ##   (9962 observations deleted due to missingness)
    ## Multiple R-squared:  0.009974,   Adjusted R-squared:  0.009942 
    ## F-statistic: 307.6 on 1 and 30528 DF,  p-value: < 2.2e-16

``` r
names(summary(fit))
```

    ##  [1] "call"          "terms"         "residuals"     "coefficients" 
    ##  [5] "aliased"       "sigma"         "df"            "r.squared"    
    ##  [9] "adj.r.squared" "fstatistic"    "cov.unscaled"  "na.action"

``` r
coef(fit)
```

    ## (Intercept)       stars 
    ##   -66.50023    44.11475

``` r
broom::tidy(fit) #tidys up the summary into a tibble, easily interpret. since tibble can tidy it 
```

    ## # A tibble: 2 × 5
    ##   term        estimate std.error statistic  p.value
    ##   <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    ## 1 (Intercept)    -66.5     11.9      -5.59 2.27e- 8
    ## 2 stars           44.1      2.52     17.5  1.61e-68

``` r
fit %>% 
  broom::tidy() %>% 
  select(term, estimate, p.value)
```

    ## # A tibble: 2 × 3
    ##   term        estimate  p.value
    ##   <chr>          <dbl>    <dbl>
    ## 1 (Intercept)    -66.5 2.27e- 8
    ## 2 stars           44.1 1.61e-68

``` r
fit %>% 
  broom::glance()
```

    ## # A tibble: 1 × 12
    ##   r.squared adj.r.squared sigma statistic  p.value    df   logLik     AIC    BIC
    ##       <dbl>         <dbl> <dbl>     <dbl>    <dbl> <dbl>    <dbl>   <dbl>  <dbl>
    ## 1   0.00997       0.00994  184.      308. 1.61e-68     1 -202491. 404989. 4.05e5
    ## # ℹ 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

Try a bit more complex

``` r
fit = 
  lm(price ~ stars + borough, data = nyc_airbnb)
```

borough broken up into one reference category (the bronx) and three
remaining boroughs are compared to the bronx (ordered based
alphabetical)

``` r
fit = 
  lm(price ~ stars + borough, data = nyc_airbnb)

fit %>% 
  broom::tidy() %>% 
  select(term, estimate, p.value) %>% 
  mutate(
    term = str_replace(term, "borough", "Borough: ")
  ) %>% 
  knitr::kable(digits = 3)
```

| term              | estimate | p.value |
|:------------------|---------:|--------:|
| (Intercept)       |   19.839 |   0.104 |
| stars             |   31.990 |   0.000 |
| Borough: Brooklyn |  -49.754 |   0.000 |
| Borough: Queens   |  -77.048 |   0.000 |
| Borough: Bronx    |  -90.254 |   0.000 |

But then change with new order of the nyc_airbnb data we factored/
reordering the borough based on frequency, so Manhatten became reference

But still can compare Brooklyn and Queens by just comparing the two
intercepts. Nothign inherently changes, just the reference

## Some Diagnostics

``` r
nyc_airbnb %>% 
  ggplot(aes(x = stars, y = price)) + 
  geom_point() +
  stat_smooth(method = "lm")
```

    ## `geom_smooth()` using formula = 'y ~ x'

    ## Warning: Removed 9962 rows containing non-finite outside the scale range
    ## (`stat_smooth()`).

    ## Warning: Removed 9962 rows containing missing values or values outside the scale range
    ## (`geom_point()`).

![](20241107_linear_models_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Does the stars go up as the price goes up? Looking at this plot this is
about right (but there are extreme outliers; residuals)

Regression diagnostics can identify issues in model fit, especially
related to certain failures in model assumptions. Examining residuals
and fitted values are therefore an imporant component of any modeling
exercise.

Most diagnostics use residuals

``` r
modelr::add_residuals(nyc_airbnb, fit)
```

    ## # A tibble: 40,492 × 6
    ##    price stars borough neighborhood room_type        resid
    ##    <dbl> <dbl> <fct>   <chr>        <fct>            <dbl>
    ##  1    99   5   Bronx   City Island  Private room      9.47
    ##  2   200  NA   Bronx   City Island  Private room     NA   
    ##  3   300  NA   Bronx   City Island  Entire home/apt  NA   
    ##  4   125   5   Bronx   City Island  Entire home/apt  35.5 
    ##  5    69   5   Bronx   City Island  Private room    -20.5 
    ##  6   125   5   Bronx   City Island  Entire home/apt  35.5 
    ##  7    85   5   Bronx   City Island  Entire home/apt  -4.53
    ##  8    39   4.5 Bronx   Allerton     Private room    -34.5 
    ##  9    95   5   Bronx   Allerton     Entire home/apt   5.47
    ## 10   125   4.5 Bronx   Allerton     Entire home/apt  51.5 
    ## # ℹ 40,482 more rows

Now have a dataframe and can do dataframe things to it

``` r
modelr::add_residuals(nyc_airbnb, fit) %>% 
  ggplot(aes(x = resid)) +
  geom_histogram()
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

    ## Warning: Removed 9962 rows containing non-finite outside the scale range
    ## (`stat_bin()`).

![](20241107_linear_models_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
modelr::add_residuals(nyc_airbnb, fit) %>% 
  ggplot(aes(x = borough, y = resid)) +
  geom_violin() + 
  ylim(-200, 200)
```

    ## Warning: Removed 11208 rows containing non-finite outside the scale range
    ## (`stat_ydensity()`).

![](20241107_linear_models_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

See that residuals are relatively skewed and the residual distribution
different across different boroughs

This gives us an understanding of how our model regression is fitting
our data. Whether there is bias, fitting one borough better than another
etc.

Residuals against star ratings

``` r
modelr::add_residuals(nyc_airbnb, fit) %>% # in this line of code, use modelr fit this dataframe and use fit to add residuals
  ggplot(aes(x = stars, y = resid)) +
  geom_point() 
```

    ## Warning: Removed 9962 rows containing missing values or values outside the scale range
    ## (`geom_point()`).

![](20241107_linear_models_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Can compare residuals against fitted values

``` r
nyc_airbnb %>% 
  modelr::add_residuals(fit) %>% 
  modelr::add_predictions(fit) %>% 
  ggplot(aes(x = pred, y = resid)) + 
  geom_point()
```

    ## Warning: Removed 9962 rows containing missing values or values outside the scale range
    ## (`geom_point()`).

![](20241107_linear_models_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->
