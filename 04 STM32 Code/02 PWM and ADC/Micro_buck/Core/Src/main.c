/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc2;
ADC_HandleTypeDef hadc5;
DMA_HandleTypeDef hdma_adc2;
DMA_HandleTypeDef hdma_adc5;

TIM_HandleTypeDef htim1;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
#define UART_TRANSMISSION_INTERVAL_MS 100
#define UART_TRANSMISSION_TIMEOUT_ms 5
#define SEND_BINARY_ADC_READINGS 0

#define ADC2_BUF_LEN 2
#define ADC5_BUF_LEN 2

volatile uint16_t adc2_buffer[ADC2_BUF_LEN];
volatile uint16_t adc5_buffer[ADC5_BUF_LEN];

static const float Vin_rdiv_factor = (50.0f + 1.65f) / 1.65f;  # Resistive divider
static const float Iin_rs_factor   = 1.0f / (0.001f/2.0f * 8.2f * 10.0f);  #1/(Rs*k_amc*k_oa)
static const float Vo_rdiv_factor  = (50.0f + 2.7f) / 2.7f;  # Resistive divider
static const float Io_rs_factor    = 1.0f / (0.001f/2.0f * 8.2f * 17.0f);  #1/(Rs*k_amc*k_oa)

volatile float Vin_V = 0.0f;
volatile float Vout_V = 0.0f;
volatile float Iin_A = 0.0f;
volatile float Io_A = 0.0f;

volatile uint16_t Vin_raw = 0.0;
volatile uint16_t Vout_raw = 0.0;
volatile uint16_t Iin_raw = 0.0;
volatile uint16_t Io_raw = 0.0;

char tx_buffer[120];
uint32_t last_tx_time = 0;

volatile uint8_t mode_bo_active = 1;
volatile uint8_t mode_dpt_periodic = 1;

#define CMD_BUFFER_SIZE 64
char rx_cmd_buffer[CMD_BUFFER_SIZE];
uint16_t rx_index = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC2_Init(void);
static void MX_ADC5_Init(void);
static void MX_TIM1_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

void UART_CheckForIncomingCommands(void);
void Parse_SerialCommand(char *cmd_str);
void PWM_SetOutputState(uint8_t enabled);
void PWM_SetDutyCycle(float duty_percent);
void PWM_SetFrequency(uint32_t frequency_hz);
void PWM_SetDeadTime_ns(uint32_t deadtime_ns);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC2_Init();
  MX_ADC5_Init();
  MX_TIM1_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */


  HAL_ADCEx_Calibration_Start(&hadc2, ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc5, ADC_SINGLE_ENDED);

  HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc2_buffer, ADC2_BUF_LEN);
  HAL_ADC_Start_DMA(&hadc5, (uint32_t*)adc5_buffer, ADC5_BUF_LEN);

  // Default-safe state: BO PWM stays off until explicitly enabled by serial command.
  PWM_SetOutputState(0);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  // Check for and process incoming serial commands from Python GUI
	  UART_CheckForIncomingCommands();

	  if (SEND_BINARY_ADC_READINGS){
		  // integer/bitshift efficient operation (1LSB error due to 4095/4096)
		  Iin_raw = adc2_buffer[0];  // PA5
		  Io_raw = adc2_buffer[1];  // PA6
		  Vin_raw = adc5_buffer[0];  // PA8
		  Vout_raw = adc5_buffer[1];  // PA9

	  }
	  else{
		  // float implementation
		  Iin_A = (adc2_buffer[0] * 3.3f) / 4095.0f * Iin_rs_factor;   // PA5
		  Io_A = (adc2_buffer[1] * 3.3f) / 4095.0f * Io_rs_factor;  // PA6
		  Vin_V= (adc5_buffer[0] * 3.3f) / 4095.0f * Vin_rdiv_factor;   // PA8
		  Vout_V = (adc5_buffer[1] * 3.3f) / 4095.0f * Vo_rdiv_factor;    // PA9
	  }


  	  if (HAL_GetTick() - last_tx_time >= UART_TRANSMISSION_INTERVAL_MS)
  	  {
  		  last_tx_time = HAL_GetTick();
  		  int sprint_len = 0;
  		  if (SEND_BINARY_ADC_READINGS){
  			  sprint_len = snprintf(tx_buffer, sizeof(tx_buffer),
			  "Vin:%ib Vo:%ib Iin:%ib Io:%ib\r\n",
			  Vin_raw, Vout_raw, Iin_raw, Io_raw);
  		  }
  		  else{
  			  sprint_len = snprintf(tx_buffer, sizeof(tx_buffer),
			  "Vin:%.3fV Vo:%.3fV Iin:%.3fA Io:%.3fA\r\n",
			  Vin_V, Vout_V, Iin_A, Io_A);
  		  }
  		  if (sprint_len > 0){
  			  HAL_UART_Transmit(&huart2, (uint8_t*)tx_buffer, (uint16_t)sprint_len, UART_TRANSMISSION_TIMEOUT_ms);
  		  }
  	  }

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV4;
  RCC_OscInitStruct.PLL.PLLN = 85;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC2_Init(void)
{

  /* USER CODE BEGIN ADC2_Init 0 */

  /* USER CODE END ADC2_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC2_Init 1 */

  /* USER CODE END ADC2_Init 1 */

  /** Common config
  */
  hadc2.Instance = ADC2;
  hadc2.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc2.Init.Resolution = ADC_RESOLUTION_12B;
  hadc2.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc2.Init.GainCompensation = 0;
  hadc2.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc2.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc2.Init.LowPowerAutoWait = DISABLE;
  hadc2.Init.ContinuousConvMode = ENABLE;
  hadc2.Init.NbrOfConversion = 2;
  hadc2.Init.DiscontinuousConvMode = DISABLE;
  hadc2.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc2.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc2.Init.DMAContinuousRequests = ENABLE;
  hadc2.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc2.Init.OversamplingMode = ENABLE;
  hadc2.Init.Oversampling.Ratio = ADC_OVERSAMPLING_RATIO_16;
  hadc2.Init.Oversampling.RightBitShift = ADC_RIGHTBITSHIFT_4;
  hadc2.Init.Oversampling.TriggeredMode = ADC_TRIGGEREDMODE_SINGLE_TRIGGER;
  hadc2.Init.Oversampling.OversamplingStopReset = ADC_REGOVERSAMPLING_CONTINUED_MODE;
  if (HAL_ADC_Init(&hadc2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_13;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_47CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_3;
  sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC2_Init 2 */

  /* USER CODE END ADC2_Init 2 */

}

/**
  * @brief ADC5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC5_Init(void)
{

  /* USER CODE BEGIN ADC5_Init 0 */

  /* USER CODE END ADC5_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC5_Init 1 */

  /* USER CODE END ADC5_Init 1 */

  /** Common config
  */
  hadc5.Instance = ADC5;
  hadc5.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc5.Init.Resolution = ADC_RESOLUTION_12B;
  hadc5.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc5.Init.GainCompensation = 0;
  hadc5.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc5.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc5.Init.LowPowerAutoWait = DISABLE;
  hadc5.Init.ContinuousConvMode = ENABLE;
  hadc5.Init.NbrOfConversion = 2;
  hadc5.Init.DiscontinuousConvMode = DISABLE;
  hadc5.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc5.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc5.Init.DMAContinuousRequests = ENABLE;
  hadc5.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc5.Init.OversamplingMode = ENABLE;
  hadc5.Init.Oversampling.Ratio = ADC_OVERSAMPLING_RATIO_16;
  hadc5.Init.Oversampling.RightBitShift = ADC_RIGHTBITSHIFT_4;
  hadc5.Init.Oversampling.TriggeredMode = ADC_TRIGGEREDMODE_SINGLE_TRIGGER;
  hadc5.Init.Oversampling.OversamplingStopReset = ADC_REGOVERSAMPLING_CONTINUED_MODE;
  if (HAL_ADC_Init(&hadc5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_1;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_47CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc5, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_2;
  sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc5, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC5_Init 2 */

  /* USER CODE END ADC5_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 1699;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 850;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 9;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.BreakFilter = 0;
  sBreakDeadTimeConfig.BreakAFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.Break2State = TIM_BREAK2_DISABLE;
  sBreakDeadTimeConfig.Break2Polarity = TIM_BREAK2POLARITY_HIGH;
  sBreakDeadTimeConfig.Break2Filter = 0;
  sBreakDeadTimeConfig.Break2AFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart2, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart2, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMAMUX1_CLK_ENABLE();
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);
  /* DMA1_Channel2_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel2_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel2_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void PWM_SetOutputState(uint8_t enabled)
{
  if (enabled)
  {
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
    HAL_TIMEx_PWMN_Start(&htim1, TIM_CHANNEL_3);
    __HAL_TIM_MOE_ENABLE(&htim1);
  }
  else
  {
    __HAL_TIM_MOE_DISABLE(&htim1);
    HAL_TIMEx_PWMN_Stop(&htim1, TIM_CHANNEL_3);
    HAL_TIM_PWM_Stop(&htim1, TIM_CHANNEL_3);
  }
}

void PWM_SetDutyCycle(float duty_percent)
{
    // Bound the input to safe limits
    if (duty_percent < 0.0f) duty_percent = 0.0f;
    if (duty_percent > 100.0f) duty_percent = 100.0f;

    // Get current ARR max value
    uint32_t current_arr = __HAL_TIM_GET_AUTORELOAD(&htim1);

    // Calculate new pulse value based on ARR scale
    uint32_t pulse = (uint32_t)((duty_percent / 100.0f) * (float)(current_arr + 1));

    // Write directly to the Channel 3 Compare Register
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, pulse);
}

void PWM_SetFrequency(uint32_t frequency_hz)
{
    if (frequency_hz == 0) return;

    uint32_t tim_clk = 170000000; // Your clock is explicitly configured to 170MHz
    uint32_t new_arr = (tim_clk / frequency_hz) - 1;

    // Constrain to 16-bit timer limits (0 to 65535)
    if (new_arr > 65535) new_arr = 65535;
    if (new_arr < 1) new_arr = 1;

    // Calculate previous duty ratio to preserve it across the frequency shift
    uint32_t old_arr = __HAL_TIM_GET_AUTORELOAD(&htim1);
    uint32_t old_pulse = __HAL_TIM_GET_COMPARE(&htim1, TIM_CHANNEL_3);
    float duty_ratio = (float)old_pulse / (float)(old_arr + 1);

    // Update the Frequency window
    __HAL_TIM_SET_AUTORELOAD(&htim1, new_arr);

    // Re-apply scaled pulse value for the new window scale
    uint32_t new_pulse = (uint32_t)(duty_ratio * (float)(new_arr + 1));
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, new_pulse);
}

void PWM_SetDeadTime_ns(uint32_t deadtime_ns)
{
    // 1 tick = 1000 / 170 MHz = 5.88235 ns.
    // Convert target nanoseconds to required clock ticks, rounding up for safety.
	uint32_t ticks = ((deadtime_ns * 170) + 999) / 1000;

    // Safety clamp: 127 ticks is the maximum limit for Range 1 (~747 ns)
	// But we clamp here at 32/170MHz = 188ns to prevent errors.
    if (ticks > 32) ticks = 32;

    uint8_t dtg_value = (uint8_t)ticks;

    // Safely write to the BDTR register without disturbing other system flags
    uint32_t temp_bdtr = htim1.Instance->BDTR;
    temp_bdtr &= ~TIM_BDTR_DTG;       // Clear the current 8-bit DTG configuration
    temp_bdtr |= (uint32_t)dtg_value; // Inject the newly calculated step count
    htim1.Instance->BDTR = temp_bdtr;  // Commit back to the hardware register
}

/**
  * @brief  Parses a completed command string.
  *
  * Supported formats:
  *   PWM:DUTY:<float>          - Set duty cycle in percent (0..100)
  *   PWM:FREQ:<uint_hz>        - Set switching frequency in Hz
  *   PWM:DT:<uint_ns>          - Set deadtime in nanoseconds
  *   BO:ENABLE:<0|1>           - Enable (1) or disable (0) PWM output
  *   DPT:SINGLE_SHOT           - Trigger one DPT pulse sequence
  *   DPT:SET;<k=v>;...         - Configure DPT parameters (semicolon-separated key=value pairs)
  *     Keys: ton1_us, toff_us, ton2_us, cooldown_us, mode, period_ms, periodic_enable
  *   MODE:BO:active            - Switch MCU to Buck Operation mode
  *   MODE:DPT:<periodic|single_shot> - Switch MCU to DPT mode
  */
void Parse_SerialCommand(char *cmd_str)
{
    char *mode  = strtok(cmd_str, ":");
    char *param = strtok(NULL, ":");
    char *value = strtok(NULL, ":");

    if (mode == NULL || param == NULL) return;

    /* ---- PWM controls: duty, frequency, deadtime ---- */
    if (strcmp(mode, "PWM") == 0)
    {
        if (value == NULL) return;

        if (strcmp(param, "DUTY") == 0)
        {
            PWM_SetDutyCycle(strtof(value, NULL));
        }
        else if (strcmp(param, "FREQ") == 0)
        {
            PWM_SetFrequency((uint32_t)atoi(value));
        }
        else if (strcmp(param, "DT") == 0)
        {
            PWM_SetDeadTime_ns((uint32_t)atoi(value));
        }
    }
    /* ---- Buck operation enable/disable ---- */
    else if (strcmp(mode, "BO") == 0)
    {
        if ((strcmp(param, "ENABLE") == 0 || strcmp(param, "STATE") == 0) && value != NULL)
        {
        PWM_SetOutputState((uint8_t)(atoi(value) == 1));
        }
    }
    /* ---- DPT controls ---- */
    else if (strcmp(mode, "DPT") == 0)
    {
        if (strcmp(param, "SINGLE_SHOT") == 0)
        {
            // TODO: trigger single-shot DPT pulse sequence
        }
        else if (strncmp(param, "SET", 3) == 0 && param[3] == ';')
        {
            // Format: DPT:SET;ton1_us=X;toff_us=X;ton2_us=X;cooldown_us=X;
            //                mode=X;period_ms=X;periodic_enable=X
            char *kv_pair = strtok(param + 4, ";");
            while (kv_pair != NULL)
            {
                char *eq = strchr(kv_pair, '=');
                if (eq != NULL)
                {
                    *eq = '\0';
                    char *key = kv_pair;
                    char *val = eq + 1;
                    (void)val;

                    if      (strcmp(key, "ton1_us") == 0)         { /* TODO: set T1 */ }
                    else if (strcmp(key, "toff_us") == 0)         { /* TODO: set Toff */ }
                    else if (strcmp(key, "ton2_us") == 0)         { /* TODO: set T2 */ }
                    else if (strcmp(key, "cooldown_us") == 0)     { /* TODO: set cooldown */ }
                    else if (strcmp(key, "period_ms") == 0)       { /* TODO: set period */ }
                    else if (strcmp(key, "periodic_enable") == 0) { /* TODO: enable/disable periodic */ }
                    else if (strcmp(key, "mode") == 0)            { /* TODO: set DPT mode (periodic/single_shot) */ }
                }
                kv_pair = strtok(NULL, ";");
            }
        }
    }
    /* ---- Operation mode switch ---- */
    else if (strcmp(mode, "MODE") == 0)
    {
      if (strcmp(param, "BO") == 0)
      {
        mode_bo_active = 1;
        // Entering BO mode always starts with PWM disabled; BO:ENABLE:1 must arm it.
        PWM_SetOutputState(0);
      }
      else if (strcmp(param, "DPT") == 0)
      {
        mode_bo_active = 0;

        if (value != NULL)
        {
          if (strcmp(value, "periodic") == 0)
          {
            mode_dpt_periodic = 1;
          }
          else if (strcmp(value, "single_shot") == 0)
          {
            mode_dpt_periodic = 0;
          }
        }
      }
    }
}

/**
  * @brief  Polls the UART for incoming bytes non-blockingly, aggregates them
  * into a line buffer, and passes them to the parser when a newline arrives.
  */
void UART_CheckForIncomingCommands(void)
{
    uint8_t incoming_byte;

    // Call HAL UART receive with a timeout of 0.
    // This makes it completely non-blocking; it grabs a byte if it's there, or returns immediately.
    while (HAL_UART_Receive(&huart2, &incoming_byte, 1, 0) == HAL_OK)
    {
        // Look for string termination characters (newline or carriage return)
        if (incoming_byte == '\n' || incoming_byte == '\r')
        {
            if (rx_index > 0) // Only parse if we actually collected characters
            {
                rx_cmd_buffer[rx_index] = '\0'; // Properly null-terminate the C-string
                Parse_SerialCommand(rx_cmd_buffer); // Hand it over to the parser
                rx_index = 0; // Reset index for the next incoming command
            }
        }
        else
        {
            // Drop any invisible styling or padding characters spaces
            if (incoming_byte >= 32 && incoming_byte <= 126)
            {
                // Store the character if there is room remaining in the buffer
                if (rx_index < (CMD_BUFFER_SIZE - 1)) {
                    rx_cmd_buffer[rx_index++] = (char)incoming_byte;
                } else {
                    rx_index = 0; // Buffer overflow safety reset
                }
            }
        }
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
