# app.R  CatBoost 4分类预测网页 Shiny
options(encoding = "UTF-8")
library(shiny)
library(catboost)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# 加载模型
Catboost_model <- catboost.load_model("catboost_model.cbm")

feature_names <- c("age","ALT","ALB","DBIL","IBIL","TG","HDL","WBC","PLT","PCT","Fatpercentage","WC","HR","MIN","BMR","AST","ALP","GGT","TC","TBA","LPA","HB","RBC","Weight","BMI","TBIL","LDL")

ui <- fluidPage(
  titlePanel("CatBoost模型预测计算器"),
  sidebarLayout(
    sidebarPanel(
      width = 4,
      h4("请输入各项指标"),
      numericInput("age", "年龄 age", value = NA),
      numericInput("ALT", "ALT", value = NA),
      numericInput("ALB", "ALB", value = NA),
      numericInput("DBIL", "DBIL", value = NA),
      numericInput("IBIL", "IBIL", value = NA),
      numericInput("TG", "TG", value = NA),
      numericInput("HDL", "HDL", value = NA),
      numericInput("WBC", "WBC", value = NA),
      numericInput("PLT", "PLT", value = NA),
      numericInput("PCT", "PCT", value = NA),
      numericInput("Fatpercentage", "脂肪百分比 Fatpercentage", value = NA),
      numericInput("WC", "腰围 WC", value = NA),
      numericInput("HR", "心率 HR", value = NA),
      numericInput("MIN", "MIN", value = NA),
      numericInput("BMR", "基础代谢 BMR", value = NA),
      numericInput("AST", "AST", value = NA),
      numericInput("ALP", "ALP", value = NA),
      numericInput("GGT", "GGT", value = NA),
      numericInput("TC", "TC", value = NA),
      numericInput("TBA", "TBA", value = NA),
      numericInput("LPA", "LPA", value = NA),
      numericInput("HB", "HB", value = NA),
      numericInput("RBC", "RBC", value = NA),
      numericInput("Weight", "体重 Weight", value = NA),
      numericInput("BMI", "BMI", value = NA),
      numericInput("TBIL", "TBIL", value = NA),
      numericInput("LDL", "LDL", value = NA),
      actionButton("predict_btn", "开始预测", class = "btn-primary")
    ),
    mainPanel(
      width = 8,
      h4("预测结果（4分类概率）"),
      tableOutput("pred_table"),
      plotOutput("pred_plot", height = "500px")
    )
  )
)

server <- function(input, output) {
  pred_data <- eventReactive(input$predict_btn, {
    req(
      input$age,input$ALT,input$ALB,input$DBIL,input$IBIL,
      input$TG,input$HDL,input$WBC,input$PLT,input$PCT,
      input$Fatpercentage,input$WC,input$HR,input$MIN,input$BMR,
      input$AST,input$ALP,input$GGT,input$TC,input$TBA,
      input$LPA,input$HB,input$RBC,input$Weight,input$BMI,
      input$TBIL,input$LDL
    )
    df <- data.frame(
      age = input$age,ALT = input$ALT,ALB = input$ALB,DBIL = input$DBIL,IBIL = input$IBIL,
      TG = input$TG,HDL = input$HDL,WBC = input$WBC,PLT = input$PLT,PCT = input$PCT,
      Fatpercentage = input$Fatpercentage,WC = input$WC,HR = input$HR,MIN = input$MIN,BMR = input$BMR,
      AST = input$AST,ALP = input$ALP,GGT = input$GGT,TC = input$TC,TBA = input$TBA,
      LPA = input$LPA,HB = input$HB,RBC = input$RBC,Weight = input$Weight,BMI = input$BMI,
      TBIL = input$TBIL,LDL = input$LDL
    )
    df
  })
  
  pred_result <- reactive({
    newdata <- pred_data()
    prob <- catboost.predict(Catboost_model, newdata, prediction_type = "Probability")
    res <- data.frame(
      Category = c("Class0","Class1","Class2","Class3"),
      Probability = round(as.numeric(prob),4)
    )
    res
  })
  
  output$pred_table <- renderTable({
    pred_result()
  })
  
  output$pred_plot <- renderPlot({
    df <- pred_result()
    ggplot(df, aes(x = Category, y = Probability, fill = Category)) +
      geom_col(width = 0.6) +
      geom_text(aes(label = percent(Probability, accuracy = 0.1)), vjust = -0.3, size = 5) +
      scale_y_continuous(limits = c(0,1)) +
      scale_fill_brewer(palette = "Set2") +
      theme_bw(base_size = 14) +
      labs(x = "结局分类", y = "预测概率", title = "各分类预测概率分布") +
      theme(plot.title = element_text(hjust = 0.5))
  })
}

shinyApp(ui = ui, server = server)
