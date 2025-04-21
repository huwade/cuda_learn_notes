# 概念背景：Default Stream

在 CUDA 中，Default Stream 是所有未指定 Stream 的操作（例如 Kernel 啟動或 cudaMemcpy）進行執行的預設上下文。

核心特性
Legacy Default Stream 行為：
預設流具有特殊的同步行為。
所有活動，無論屬於哪一個 Stream，自動會與 Default Stream 的操作發生同步。
Default Stream 的用途：
它在程序簡化上非常方便（因不需要管理自己的 Stream），但在並行化場景中可能會阻礙性能最大化。
投影片的重點
投影片對 Default Stream 的作用和行為進行了視覺化展示，並指出使用 Default Stream 的潛在限制。

投影片分析

1. Default Stream 的行為描述
Kernel 或 cudaMemcpy
Description: 如果 Kernel 啟動 (<<<...>>>) 或資料傳輸 (如 cudaMemcpy) 沒有顯式指定 Stream，這些操作就由 Default Stream 管理。
不指定 Stream 時，使用 Default Stream：
cpp
執行程式碼
複製程式碼
cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice); // 使用 Default Stream
kernel<<<dimGrid, dimBlock>>>(d_x);                // 使用 Default Stream
Legacy Default Stream 行為 (同步問題)
同步性 (Synchronizing on the device)：

Default Stream 的操作對其他 Stream 有同步影響。
所有在 Default Stream 中的操作必須等待其他 Streams 的操作完成後才能開始執行。
所有其他 Streams 的操作會等 Default Stream 完成後再繼續執行。
這導致 Default Stream 行為比較阻塞，可能削弱多 Stream 的並行優勢。
投影片圖解行為展示：

Stream 1 和 Stream 2 中的操作被 Default Stream 的操作所同步，出現阻塞，延遲性的紅線直觀地顯示因同步而帶來的效率損失。
紅線代表：
Default Stream 中的操作阻塞了其他 Stream 的執行過程。
2. Default Stream 的同步行為具體描述
所有在 Default Stream 中的操作：

'''cpp
cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice); // Default Stream
kernel<<<grid, block>>>(d_x);                     // Default Stream
'''
這些操作是同步的，會阻止其他 Streams 的行程。
所有其他 Streams：

必須等待 Default Stream 的操作完成後才能繼續。
所有 Host Thread：

CUDA 主機執行緒共享同一 Default Stream，這導致所有主機線程的行程都因同步行為被影響。
3. 使用 Default Stream 的挑戰
不建議在多 Stream 高併發場景使用 Default Stream：
在高併行場景中使用 Default Stream，會因其同步性導致性能低效。
原因：
Default Stream 的同步行為會干擾不同 Streams 的並行執行，削弱 GPU 的資源利用率。
4. 將 Default Stream 轉換為 "普通流"
如果需要避免 Default Stream 的同步特性：

可以修改行為，不再共享 Legacy Default Stream。
使用以下指令：
bash
複製程式碼
nvcc --default-stream per-thread
行為轉變：
每個 Host Thread 都會分配自己的 Default Stream，不再彼此阻塞。
Default Stream 的執行情況更接近 "普通流"。
