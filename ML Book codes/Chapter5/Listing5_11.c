  /* USER CODE BEGIN 2 */
	int arr_size = 64; // Must be 32,64,128,..., 4096
	float **acc_data = get_data(arr_size);
	float fft_output[3][arr_size];
	float fft_ssd[3];
	float fft_abs[3][arr_size];
	float sma_x = 0, sma_y = 0, sma_z = 0, sma;
	float x_mean = 0, y_mean, z_mean;
	int x_pos = 0, y_pos = 0, z_pos = 0;
	for(int i = 0; i < arr_size; i++){
		x_mean += acc_data[0][i];
		y_mean += acc_data[1][i];
		z_mean += acc_data[2][i];
		x_pos += acc_data[0][i] > 0;
		y_pos += acc_data[1][i] > 0;
		z_pos += acc_data[2][i] > 0;

	}
	arm_rfft_fast_instance_f32 fft;
	arm_status res = arm_rfft_fast_init_f32(&fft, arr_size);
	if (res != 0){
      printf("FFT failed. Exiting...\n");
		exit(1);
	}

	for(int i = 0; i < 3; i++){
		float *fft_input = acc_data[i];
		arm_rfft_fast_f32(&fft, fft_input, fft_output[i], 0);
		arm_std_f32(fft_output[i], arr_size / 2, &fft_ssd[i]);
		arm_abs_f32(fft_output[i],fft_abs[i], arr_size);
	}

	for(int i = 0; i < arr_size; i++){
		sma_x += fft_abs[0][i];
		sma_y += fft_abs[1][i];
		sma_z += fft_abs[2][i];
	}

	sma = (sma_x + sma_y + sma_z) / arr_size;

	free(acc_data);
  /* USER CODE END 2 */