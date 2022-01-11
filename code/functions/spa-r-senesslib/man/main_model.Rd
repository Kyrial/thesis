\name{main_model}
\alias{main_model}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
main_model(model_name, bdds, weights, metrics)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{model_name}{
%%     ~~Describe \code{model_name} here~~
}
  \item{bdds}{
%%     ~~Describe \code{bdds} here~~
}
  \item{weights}{
%%     ~~Describe \code{weights} here~~
}
  \item{metrics}{
%%     ~~Describe \code{metrics} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (model_name, bdds, weights, metrics) 
{
    for (bdd in bdds) {
        for (weight in weights) {
            for (metric in metrics) {
                if (model_name == "VGG16") {
                  if (weight == "imagenet") {
                    layers = c("input_1", "block1_conv1", "block1_conv2", 
                      "block1_pool", "block2_conv1", "block2_conv2", 
                      "block2_pool", "block3_conv1", "block3_conv2", 
                      "block3_conv3", "block3_pool", "block4_conv1", 
                      "block4_conv2", "block4_conv3", "block4_pool", 
                      "block5_conv1", "block5_conv2", "block5_conv3", 
                      "block5_pool", "flatten", "fc1", "fc2")
                  }
                  else if (weight == "vggface") {
                    layers = c("input_1", "conv1_1", "conv1_2", 
                      "pool1", "conv2_1", "conv2_2", "pool2", 
                      "conv3_1", "conv3_2", "conv3_3", "pool3", 
                      "conv4_1", "conv4_2", "conv4_3", "pool4", 
                      "conv5_1", "conv5_2", "conv5_3", "pool5", 
                      "flatten", "fc6/relu", "fc7/relu")
                  }
                  else if (weight == "vggplaces") {
                    layers = c("input_1", "block1_conv1", "block1_conv2", 
                      "block1_pool", "block2_conv1", "block2_conv2", 
                      "block2_pool", "block3_conv1", "block3_conv2", 
                      "block3_conv3", "block3_pool", "block4_conv1", 
                      "block4_conv2", "block4_conv3", "block4_pool", 
                      "block5_conv1", "block5_conv2", "block5_conv3", 
                      "block5_pool", "flatten", "fc1", "fc2")
                  }
                }
                if (bdd == "CFD") {
                  labels_path = "../../data/redesigned/CFD/labels_CFD.csv"
                  log_path = "../../results/CFD/log_"
                }
                else if (bdd == "JEN") {
                  labels_path = "../../data/redesigned/JEN/labels_JEN.csv"
                  log_path = "../../results/JEN/log_"
                }
                else if (bdd == "SCUT-FBP") {
                  labels_path = "../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv"
                  log_path = "../../results/SCUT-FBP/log_"
                }
                else if (bdd == "MART") {
                  labels_path = "../../data/redesigned/MART/labels_MART.csv"
                  log_path = "../../results/MART/log_"
                }
                else if (bdd == "SMALLTEST") {
                  labels_path = "../../data/redesigned/small_test/labels_test.csv"
                  log_path = "../../results/smalltest/log_"
                }
                else if (bdd == "BIGTEST") {
                  labels_path = "../../data/redesigned/big_test/labels_bigtest.csv"
                  log_path = "../../results/bigtest/log_"
                }
                matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path, 
                  "_", bdd, "_", weight, "_", metric, "_", "_BRUTMETRICS", 
                  ".csv", sep = ""), simplify = FALSE))
                colnames(matrix_metrics)[2] <- "input_1"
                matrix_complexity <- do.call(cbind, fromJSON(file = paste(log_path, 
                  "_", bdd, "_", weight, "_", "mean", "_", "_BRUTMETRICS", 
                  ".csv", sep = ""), simplify = FALSE))
                colnames(matrix_complexity)[2] <- "input_1"
                if (metric \%in\% c("kurtosis", "L0")) {
                  df_metrics <- as.data.frame(matrix_metrics, 
                    optional = TRUE)
                  flatten_df_metrics = flatten(df_metrics[, -1])
                  vmin = min(unlist(flatten_df_metrics))
                  vmax = max(unlist(flatten_df_metrics))
                  df_metrics = sapply(df_metrics, as.numeric)
                  df_metrics[, -1] = df_metrics[, -1] - vmin
                  df_metrics[, -1] = df_metrics[, -1]/vmax
                }
                else {
                  df_metrics <- as.data.frame(matrix_metrics, 
                    optional = TRUE)
                  df_metrics = sapply(df_metrics, as.numeric)
                }
                df_complexity <- as.data.frame(matrix_complexity, 
                  optional = TRUE)
                df_complexity = sapply(df_complexity, as.numeric)
                df_metrics <- as.data.frame(df_metrics)
                df_complexity <- as.data.frame(df_complexity[, 
                  -1])
                corr_metrics <- cor(df_metrics[, -1])
                df_sq_metrics = df_metrics[, -1] * df_metrics[, 
                  -1]
                if (weight == "imagenet") {
                  df_sq_metrics = rename(df_sq_metrics, c(input_1 = "input_1_sq", 
                    block1_conv1 = "conv1_1_sq", block1_conv2 = "conv1_2_sq", 
                    block1_pool = "pool1_sq", block2_conv1 = "conv2_1_sq", 
                    block2_conv2 = "conv2_2_sq", block2_pool = "pool2_sq", 
                    block3_conv1 = "conv3_1_sq", block3_conv2 = "conv3_2_sq", 
                    block3_conv3 = "conv3_3_sq", block3_pool = "pool3_sq", 
                    block4_conv1 = "conv4_1_sq", block4_conv2 = "conv4_2_sq", 
                    block4_conv3 = "conv4_3_sq", block4_pool = "pool4_sq", 
                    block5_conv1 = "conv5_1_sq", block5_conv2 = "conv5_2_sq", 
                    block5_conv3 = "conv5_3_sq", block5_pool = "pool5_sq", 
                    flatten = "flatten_sq", fc1 = "fc6/relu_sq", 
                    fc2 = "fc7/relu_sq"))
                }
                else if (weight == "vggface") {
                  df_sq_metrics = rename(df_sq_metrics, c(input_1 = "input_1_sq", 
                    conv1_2 = "conv1_2_sq", conv1_1 = "conv1_1_sq", 
                    pool1 = "pool1_sq", conv2_1 = "conv2_1_sq", 
                    conv2_2 = "conv2_2_sq", pool2 = "pool2_sq", 
                    conv3_1 = "conv3_1_sq", conv3_2 = "conv3_2_sq", 
                    conv3_3 = "conv3_3_sq", pool3 = "pool3_sq", 
                    conv4_1 = "conv4_1_sq", conv4_2 = "conv4_2_sq", 
                    conv4_3 = "conv4_3_sq", pool4 = "pool4_sq", 
                    conv5_1 = "conv5_1_sq", conv5_2 = "conv5_2_sq", 
                    conv5_3 = "conv5_3_sq", pool5 = "pool5_sq", 
                    flatten = "flatten_sq", `fc6/relu` = "fc6/relu_sq", 
                    `fc7/relu` = "fc7/relu_sq"))
                }
                else if (weight == "vggplaces") {
                  df_sq_metrics = rename(df_sq_metrics, c(input_1 = "input_1_sq", 
                    block1_conv1 = "conv1_1_sq", block1_conv2 = "conv1_2_sq", 
                    block1_pool = "pool1_sq", block2_conv1 = "conv2_1_sq", 
                    block2_conv2 = "conv2_2_sq", block2_pool = "pool2_sq", 
                    block3_conv1 = "conv3_1_sq", block3_conv2 = "conv3_2_sq", 
                    block3_conv3 = "conv3_3_sq", block3_pool = "pool3_sq", 
                    block4_conv1 = "conv4_1_sq", block4_conv2 = "conv4_2_sq", 
                    block4_conv3 = "conv4_3_sq", block4_pool = "pool4_sq", 
                    block5_conv1 = "conv5_1_sq", block5_conv2 = "conv5_2_sq", 
                    block5_conv3 = "conv5_3_sq", block5_pool = "pool5_sq", 
                    flatten = "flatten_sq", fc1 = "fc6/relu_sq", 
                    fc2 = "fc7/relu_sq"))
                }
                df_sq_metrics <- cbind(df_metrics, df_sq_metrics)
                df_complexity = rename(df_complexity, c(input_1 = "input_1", 
                  block1_conv1 = "conv1_1", block1_conv2 = "conv1_2", 
                  block1_pool = "pool1", block2_conv1 = "conv2_1", 
                  block2_conv2 = "conv2_2", block2_pool = "pool2", 
                  block3_conv1 = "conv3_1", block3_conv2 = "conv3_2", 
                  block3_conv3 = "conv3_3", block3_pool = "pool3", 
                  block4_conv1 = "conv4_1", block4_conv2 = "conv4_2", 
                  block4_conv3 = "conv4_3", block4_pool = "pool4", 
                  block5_conv1 = "conv5_1", block5_conv2 = "conv5_2", 
                  block5_conv3 = "conv5_3", block5_pool = "pool5", 
                  flatten = "flatten", fc1 = "fc6/relu", fc2 = "fc7/relu"))
                df_complexity_metrics <- cbind(df_complexity$conv4_3, 
                  df_metrics)
                df_complexity_metrics = rename(df_complexity_metrics, 
                  c(`df_complexity$conv4_3` = "complexity"))
                df_birkhoff = df_metrics[, -1:-2]/df_complexity$conv4_3
                df_birkhoff = cbind(df_metrics$rate, df_birkhoff)
                df_birkhoff = rename(df_birkhoff, c(`df_metrics$rate` = "rate"))
                df_birkhoff2 = df_metrics[, -1:-2] * df_complexity$conv4_3
                df_birkhoff2 = cbind(df_metrics$rate, df_birkhoff2)
                df_birkhoff2 = rename(df_birkhoff2, c(`df_metrics$rate` = "rate"))
                if (weight \%in\% c("imagenet", "vggplaces")) {
                  df_metrics = rename(df_metrics, c(input_1 = "input_1", 
                    block1_conv1 = "conv1_1", block1_conv2 = "conv1_2", 
                    block1_pool = "pool1", block2_conv1 = "conv2_1", 
                    block2_conv2 = "conv2_2", block2_pool = "pool2", 
                    block3_conv1 = "conv3_1", block3_conv2 = "conv3_2", 
                    block3_conv3 = "conv3_3", block3_pool = "pool3", 
                    block4_conv1 = "conv4_1", block4_conv2 = "conv4_2", 
                    block4_conv3 = "conv4_3", block4_pool = "pool4", 
                    block5_conv1 = "conv5_1", block5_conv2 = "conv5_2", 
                    block5_conv3 = "conv5_3", block5_pool = "pool5", 
                    flatten = "flatten", fc1 = "fc6/relu", fc2 = "fc7/relu"))
                  df_complexity_metrics = rename(df_complexity_metrics, 
                    c(input_1 = "input_1", block1_conv1 = "conv1_1", 
                      block1_conv2 = "conv1_2", block1_pool = "pool1", 
                      block2_conv1 = "conv2_1", block2_conv2 = "conv2_2", 
                      block2_pool = "pool2", block3_conv1 = "conv3_1", 
                      block3_conv2 = "conv3_2", block3_conv3 = "conv3_3", 
                      block3_pool = "pool3", block4_conv1 = "conv4_1", 
                      block4_conv2 = "conv4_2", block4_conv3 = "conv4_3", 
                      block4_pool = "pool4", block5_conv1 = "conv5_1", 
                      block5_conv2 = "conv5_2", block5_conv3 = "conv5_3", 
                      block5_pool = "pool5", flatten = "flatten", 
                      fc1 = "fc6/relu", fc2 = "fc7/relu"))
                  df_birkhoff2 = rename(df_birkhoff2, c(input_1 = "input_1", 
                    block1_conv1 = "conv1_1", block1_conv2 = "conv1_2", 
                    block1_pool = "pool1", block2_conv1 = "conv2_1", 
                    block2_conv2 = "conv2_2", block2_pool = "pool2", 
                    block3_conv1 = "conv3_1", block3_conv2 = "conv3_2", 
                    block3_conv3 = "conv3_3", block3_pool = "pool3", 
                    block4_conv1 = "conv4_1", block4_conv2 = "conv4_2", 
                    block4_conv3 = "conv4_3", block4_pool = "pool4", 
                    block5_conv1 = "conv5_1", block5_conv2 = "conv5_2", 
                    block5_conv3 = "conv5_3", block5_pool = "pool5", 
                    flatten = "flatten", fc1 = "fc6/relu", fc2 = "fc7/relu"))
                  df_birkhoff = rename(df_birkhoff, c(input_1 = "input_1", 
                    block1_conv1 = "conv1_1", block1_conv2 = "conv1_2", 
                    block1_pool = "pool1", block2_conv1 = "conv2_1", 
                    block2_conv2 = "conv2_2", block2_pool = "pool2", 
                    block3_conv1 = "conv3_1", block3_conv2 = "conv3_2", 
                    block3_conv3 = "conv3_3", block3_pool = "pool3", 
                    block4_conv1 = "conv4_1", block4_conv2 = "conv4_2", 
                    block4_conv3 = "conv4_3", block4_pool = "pool4", 
                    block5_conv1 = "conv5_1", block5_conv2 = "conv5_2", 
                    block5_conv3 = "conv5_3", block5_pool = "pool5", 
                    flatten = "flatten", fc1 = "fc6/relu", fc2 = "fc7/relu"))
                }
                print(paste("parameters are:", bdd, "-", weight, 
                  "-", metric))
                df_complexity_metrics$complexity <- scale(df_complexity_metrics$complexity)
                df_complexity_metrics$input_1 <- scale(df_complexity_metrics$input_1)
                df_complexity_metrics$conv1_1 <- scale(df_complexity_metrics$conv1_1)
                df_complexity_metrics$conv1_2 <- scale(df_complexity_metrics$conv1_2)
                df_complexity_metrics$pool1 <- scale(df_complexity_metrics$pool1)
                df_complexity_metrics$conv2_1 <- scale(df_complexity_metrics$conv2_1)
                df_complexity_metrics$conv2_2 <- scale(df_complexity_metrics$conv2_2)
                df_complexity_metrics$pool2 <- scale(df_complexity_metrics$pool2)
                df_complexity_metrics$conv3_1 <- scale(df_complexity_metrics$conv3_1)
                df_complexity_metrics$conv3_2 <- scale(df_complexity_metrics$conv3_2)
                df_complexity_metrics$conv3_3 <- scale(df_complexity_metrics$conv3_3)
                df_complexity_metrics$pool3 <- scale(df_complexity_metrics$pool3)
                df_complexity_metrics$conv4_1 <- scale(df_complexity_metrics$conv4_1)
                df_complexity_metrics$conv4_2 <- scale(df_complexity_metrics$conv4_2)
                df_complexity_metrics$conv4_3 <- scale(df_complexity_metrics$conv4_3)
                df_complexity_metrics$pool4 <- scale(df_complexity_metrics$pool4)
                df_complexity_metrics$conv5_1 <- scale(df_complexity_metrics$conv5_1)
                df_complexity_metrics$conv5_2 <- scale(df_complexity_metrics$conv5_2)
                df_complexity_metrics$conv5_3 <- scale(df_complexity_metrics$conv5_3)
                df_complexity_metrics$pool5 <- scale(df_complexity_metrics$pool5)
                model_int_complexity = step(lm(rate ~ (conv1_2 + 
                  conv1_1 + pool1 + conv2_1 + conv2_2 + pool2 + 
                  conv3_1 + conv3_2 + conv3_3 + pool3 + conv4_1 + 
                  conv4_2 + conv4_3 + pool4 + conv5_1 + conv5_2 + 
                  conv5_3 + pool5 + complexity):complexity + 
                  conv1_1 + conv1_2 + pool1 + conv2_1 + conv2_2 + 
                  pool2 + conv3_1 + conv3_2 + conv3_3 + pool3 + 
                  conv4_1 + conv4_2 + conv4_3 + pool4 + conv5_1 + 
                  conv5_2 + conv5_3 + pool5, data = df_complexity_metrics), 
                  trace = 0)
                print(summary(model_int_complexity))
            }
        }
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
