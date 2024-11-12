20241112_cross_validation
================
2024-11-12

Question: which linear regression will you use? Useful when there is one
super clear hypothesis and want to know whether this one variable
important in the context of the overall analysis

But more than often, need to assess what best fits into the model and
compare models, but have to be worried about multiple comparisons
(fishing)

Basic question of modelling:

- does my model fit and is it too complex? (under / over fitting)

- do i have high bias (just off from the truth) or high variance (one
  dataset get this answer another dataset will get another answer)

- will my model generalise to future datasets? is what im doing only
  depenedent on the sample that i have now or can it work well with the
  future too? Balancing underfit to overfit

**Model selection**

to see whether the model that i built on my first dataset will be
accurate for a second dataset (whether the prediction can work); but
this is rare that someone will give you a new dataset to just make
predictions (usually only get one dataset)

To overcome this, split the one dataset into two (training 80% and
testing 20%)

- Training data is what used to build the model
- Testing is the data that you use to evaluate out-of-sample fit

Evaluate the model (accurate of predictions) with root mean squared
error (lowest)

**Refinements and variations**

repeat this model building (repeat the training and tetsing split)

**k-fold validation**

instead splitting into 80/20 training split but you partition the data
into k equally sized subsets

**Prediction rather than statistical inference**

Which model is going to be the most generalisable. Do not get a p-value
of whether my model is accurate. Means that need to think about /
evaluate my model differently from hypothesis testing when you just
compare to a p-value / null hypothesis

**Steps**

- Training and Testing split

- Generate model and get root mean squared error

- Repeat the training and testing split

- minimize root mean squared error

## Look at LIDAR data

The lidar data frame has 221 observations from a light detection and
ranging (LIDAR) experiment.

range = distance travelled before the light is reflected back to its
source.

logratio = logarithm of the ratio of received light from two laser
sources.

``` r
data("lidar")

lidar_df = 
  lidar %>% 
  as_tibble() %>% 
  mutate(
    id = row_number())
```

``` r
lidar_df %>% 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point()
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

## Try to do Cross Validation

We’ll compare 3 models – one linear, smooth and wiggly

``` r
train_df = sample_frac(lidar_df, size = .8)
#split into training data set, then testing dataset is everything that is not in the training dataset 

test_df = anti_join(lidar_df, train_df, by = "id")
```

Looking at these dataframes

``` r
ggplot(train_df, aes(x = range, y = logratio)) +
  geom_point() +
  geom_point( data = test_df, color = "red")
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Now trying to fitting models (one model is too complex another under
complex and both make bad predictions. then the last is just right)

## Fitting a models

order: not complex enough, just right, too complex

``` r
linear_mod = lm(logratio ~ range, data = train_df)
smooth_mod = gam(logratio ~ s(range), data = train_df)
wiggly_mod = gam(logratio ~ s(range, k = 30), sp = 10e-6, data = train_df)
```

Looking at these fits

``` r
train_df %>% 
  add_predictions(linear_mod) %>% 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point() + 
  geom_line(aes(y = pred), color = "red")
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
train_df %>% 
  add_predictions(wiggly_mod) %>% 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point() + 
  geom_line(aes(y = pred), color = "red")
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

This wiggly model is overcompensating, overfitting the dataset. that if
the model was applied to another dataset it won’t be able to predict so
well.

``` r
train_df %>% 
  add_predictions(smooth_mod) %>% 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point() + 
  geom_line(aes(y = pred), color = "red")
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

smooth model fit is just right, flexible model structure that it
captures the trends but it doesn’t overfit

## Compare these models numerically using RMSE

``` r
rmse(linear_mod, test_df)
```

    ## [1] 0.127317

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.08302008

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.08848557

But what do these values mean? Whether the difference between the smooth
(0.06477024) and wiggly (0.06714495) model is valid or if this is just
because of the specific training and testing split that we did

To answer this we repeat the training and testing split and create
models over and over

## Repeat the train / test split

Instead of writing function repeatedly, there is already a package made
(giving list cols, 100 rows where each row has a set of 80/20 split
dataframe)

``` r
cv_df = 
  crossv_mc(lidar_df, 100)
#by default 80/20, 100 splits
```

But crossv_mc will give us only information on which rows were split
into the training and testing section, so we need to have to get back
the information

``` r
cv_df = 
  crossv_mc(lidar_df, 100) %>% 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble))
```

``` r
cv_res_df = 
  cv_df %>% 
  mutate(
    linear_mod = map(train, \(x) lm(logratio ~ range, data = x))) %>% 
  mutate(
    rmse_linear = map2_dbl(linear_mod, test, rmse)
  )

print(cv_res_df)
```

    ## # A tibble: 100 × 5
    ##    train              test              .id   linear_mod rmse_linear
    ##    <list>             <list>            <chr> <list>           <dbl>
    ##  1 <tibble [176 × 3]> <tibble [45 × 3]> 001   <lm>             0.134
    ##  2 <tibble [176 × 3]> <tibble [45 × 3]> 002   <lm>             0.142
    ##  3 <tibble [176 × 3]> <tibble [45 × 3]> 003   <lm>             0.128
    ##  4 <tibble [176 × 3]> <tibble [45 × 3]> 004   <lm>             0.142
    ##  5 <tibble [176 × 3]> <tibble [45 × 3]> 005   <lm>             0.136
    ##  6 <tibble [176 × 3]> <tibble [45 × 3]> 006   <lm>             0.143
    ##  7 <tibble [176 × 3]> <tibble [45 × 3]> 007   <lm>             0.140
    ##  8 <tibble [176 × 3]> <tibble [45 × 3]> 008   <lm>             0.130
    ##  9 <tibble [176 × 3]> <tibble [45 × 3]> 009   <lm>             0.135
    ## 10 <tibble [176 × 3]> <tibble [45 × 3]> 010   <lm>             0.137
    ## # ℹ 90 more rows

repeat this process with smooth / wiggly fit

``` r
cv_res_df = 
  cv_df %>% 
  mutate(
    linear_mod = map(train, \(x) lm(logratio ~ range, data = x)),
    smooth_mod = map(train, \(x) gam(logratio ~ range, data = x)),
    wiggly_mod = map(train, \(x) gam(logratio ~ s(range, k = 30), sp = 10e-6, data = x))) %>% 
  mutate(
    rmse_linear = map2_dbl(linear_mod, test, rmse),
    rmse_smooth = map2_dbl(smooth_mod, test, rmse),
    rmse_wiggly = map2_dbl(wiggly_mod, test, rmse)
  )

#mapping is saying that i am passing linear_mod and test dataframe into the rmse function
```

## Looking now at the RMSE distribution

``` r
cv_res_df %>% 
  select(starts_with("rmse")) %>% 
  pivot_longer(
    everything(), 
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) %>% 
  ggplot(aes(x= model, y = rmse)) + 
  geom_violin()
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

## Example Child Growth

``` r
child_df = 
  read_csv("data/nepalese_children.csv") %>% 
  mutate(
    weight_ch7 = (weight > 7) * (weight - 7) #for a piece-wise model, when weight < 7 it will be 0, when weight > 7 it will be weight - 7
  )
```

    ## Rows: 2705 Columns: 5
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (5): age, sex, weight, height, armc
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
child_df %>% 
  ggplot(aes(x = weight, y = armc)) + 
  geom_point(alpha = .5)
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

Looking at different modelling approaches

``` r
linear_mod_child = lm(armc ~ weight, data = child_df)
pwl_mod_child = lm(armc ~ weight + weight_ch7, data = child_df)
smooth_mod_child = gam(armc ~ s(weight), data = child_df)
```

Looking at these models

1.  linear model

``` r
child_df %>% 
  add_predictions(linear_mod_child) %>% 
  ggplot(aes(x = weight, y = armc)) + 
  geom_point(alpha = .5) +
  geom_line(aes(y = pred, color = "red"))
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

2.  piece-wise linear model

change point at 7

``` r
child_df %>% 
  add_predictions(pwl_mod_child) %>% 
  ggplot(aes(x = weight, y = armc)) + 
  geom_point(alpha = .5) +
  geom_line(aes(y = pred, color = "red"))
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

3.  Smooth model fit

``` r
child_df %>% 
  add_predictions(smooth_mod_child) %>% 
  ggplot(aes(x = weight, y = armc)) + 
  geom_point(alpha = .5) +
  geom_line(aes(y = pred, color = "red"))
```

![](20241112_cross_validation_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->
