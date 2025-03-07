$(document).ready(function() {
    // 全局变量，用于存储开始时间
    var startTime;
    // 估计的总迭代次数 (可以根据经验调整)
    const estimatedTotalIterations = 5;  // 假设最多重写5次

    // 获取 Gemini 模型列表的函数
    function fetchGeminiModels() {
        var geminiApiKey = $('#geminiApiKey').val();
        $('#geminiModelAlert').hide(); // 隐藏之前的 Gemini 模型相关的警告

        if (!geminiApiKey) {
            $('#geminiModelAlert').text('请输入 Gemini API 密钥以获取模型列表!').show();
            return;
        }

        $.ajax({
            url: '/get_gemini_models',
            type: 'GET',
            data: { gemini_api_key: geminiApiKey },
            success: function(data) {
                $('#geminiModel').empty();
                // 先添加配置文件中的选项
                var configGeminiModel = $('#geminiModel option:selected').val();
                if (configGeminiModel) {
                    $('#geminiModel').append($('<option>', {
                        value: configGeminiModel,
                        text: configGeminiModel
                    }));
                }
                // 再添加API返回的选项
                $.each(data.models, function(index, model) {
                    // 避免重复添加
                    if (model !== configGeminiModel) {
                        $('#geminiModel').append($('<option>', {
                            value: model,
                            text: model
                        }));
                    }
                });
                $('#geminiModelAlert').text('模型列表已更新。').show(); // 显示成功消息

            },
            error: function(xhr, status, error) {
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    $('#geminiModelAlert').text(xhr.responseJSON.error).show();
                } else {
                    $('#geminiModelAlert').text('获取 Gemini 模型列表失败: ' + error).show();
                }
            }
        });
    }
    // 刷新Gemini模型列表按钮点击事件
    $('#refreshGeminiModels').click(function() {
      $(this).find('i').addClass('fa-spin'); // 添加旋转类
      fetchGeminiModels();
      $(this).find('i').removeClass('fa-spin'); //移除旋转类

    });

     // 获取 SiliconFlow 模型列表的函数
    function fetchSiliconFlowModels() {
        var siliconFlowApiKey = $('#siliconFlowApiKey').val();
        $('#siliconFlowModelAlert').hide();

        if (!siliconFlowApiKey) {
            $('#siliconFlowModelAlert').text('请输入 SiliconFlow API 密钥以获取模型列表!').show();
            return;
        }

        $.ajax({
            url: '/get_siliconflow_models',
            type: 'GET',
            data: { siliconflow_api_key: siliconFlowApiKey },
            success: function(data) {
                $('#siliconFlowModel').empty();
                // 先添加配置文件中的选项
                var configSiliconFlowModel = $('#siliconFlowModel option:selected').val();
                if(configSiliconFlowModel){
                  $('#siliconFlowModel').append($('<option>', {
                        value: configSiliconFlowModel,
                        text: configSiliconFlowModel
                    }));
                }
                $.each(data.models, function(index, model) {
                  //避免重复添加
                    if (model !== configSiliconFlowModel){
                      $('#siliconFlowModel').append($('<option>', {
                            value: model,
                            text: model
                        }));
                    }
                });
                $('#siliconFlowModelAlert').text('模型列表已更新。').show();
            },
            error: function(xhr, status, error) {
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    $('#siliconFlowModelAlert').text(xhr.responseJSON.error).show();
                } else {
                    $('#siliconFlowModelAlert').text('获取 SiliconFlow 模型列表失败: ' + error).show();
                }
            }
        });
    }

    // 刷新 SiliconFlow 模型列表按钮点击事件
    $('#refreshSiliconFlowModels').click(function() {
        $(this).find('i').addClass('fa-spin');
        fetchSiliconFlowModels();
        $(this).find('i').removeClass('fa-spin');
    });

    // AI 检测器选择改变事件
    $('#detectorSelect').change(function() {
        if ($(this).val() === 'sapling') {
            $('#saplingApiKeyGroup').show();
            $('#winstonApiKeyGroup').hide();
        } else {
            $('#saplingApiKeyGroup').hide();
            $('#winstonApiKeyGroup').show();
        }
    });

    // 提交按钮点击事件
    $('#submitBtn').click(function() {
        // 获取用户输入
        var inputText = $('#inputText').val();
        var prompt = $('#prompt').val();
        var geminiApiKey = $('#geminiApiKey').val();
        var siliconFlowApiKey = $('#siliconFlowApiKey').val();
        var detector = $('#detectorSelect').val(); // 获取选择的检测器
        var saplingApiKey = $('#saplingApiKey').val();
        var winstonApiKey = $('#winstonApiKey').val();
        var geminiModel = $('#geminiModel').val();
        var siliconFlowModel = $('#siliconFlowModel').val();
        var sentenceThreshold = parseFloat($('#sentenceThreshold').val());
        var paragraphThreshold = parseFloat($('#paragraphThreshold').val());


        // 清空之前的错误信息和结果,并显示初始文本
        $('#errorAlert').hide();
        $('#resultsContainer').empty();
        $('#resultsContainer').append(`<div class="card mt-3"><div class="card-header"><b>原始文本</b></div><div class="card-body"><p id="originalText"></p></div></div><div class="card mt-3" id="currentSentenceCard" style="display: none;"><div class="card-header"><b>当前重写句子</b></div><div class="card-body"><p id="currentSentence"></p></div></div>`);
        $('#originalText').text(inputText);


      // 显示加载动画和初始状态
        $('#resultsLoading').show();
        updateLoadingStatus('正在准备', 0); // 初始状态

      // 禁用提交按钮
        $('#submitBtn').prop('disabled', true);
        $('#submitBtnText').hide();
        $('#submitBtnSpinner').show();

        // 记录开始时间
        startTime = new Date().getTime();

        // 显示当前重写句子卡片
        $('#currentSentenceCard').show();

        // 准备 API 密钥数据
        var apiKeyData = {};
        if (detector === 'sapling') {
            apiKeyData.sapling_api_key = saplingApiKey;
        } else {
            apiKeyData.winston_api_key = winstonApiKey;
        }


        // 发送请求到后端
        $.ajax({
            url: '/process',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                input_text: inputText,
                prompt: prompt,
                gemini_api_key: geminiApiKey,
                siliconflow_api_key: siliconFlowApiKey,
                detector: detector, // 将选择的检测器发送到后端
                ...apiKeyData,  // 使用扩展运算符包含 API 密钥
                gemini_model: geminiModel,
                siliconflow_model: siliconFlowModel,
                sentence_threshold: sentenceThreshold,
                paragraph_threshold: paragraphThreshold
            }),
            success: function(data) {
                // 更新 API 用量
                $('#geminiUsage').text(data.gemini_usage);
                $('#siliconFlowUsage').text(data.siliconflow_usage);

                if (detector === 'sapling') {
                    $('#saplingUsage').text(data.sapling_usage);
                } else {
                    $('#winstonBalance').text(data.winston_balance);  // 更新 Winston 余额
                }
                
                $('#currentSentenceCard').hide(); // 隐藏

                // 显示结果
                $('#resultsContainer').empty(); // 清空之前的结果

                // 遍历 results 数组 (修改：处理重写步骤)
                $.each(data.results, function(index, result) {
                    var resultHtml = '<div class="card mt-3">';
                    resultHtml += '<div class="card-header"><b>' + result.model + '</b></div>';
                    resultHtml += '<div class="card-body">';

                    if (result.model === "重写模型") {
                        // 显示原始句子和重写后的句子
                        resultHtml += '<p><b>原始句子:</b> ' + result.original_sentence + '</p>';
                        resultHtml += '<p><b>重写后:</b> ' + result.rewritten_sentence + '</p>';
                         if (result.sapling_score !== undefined && result.sapling_score !== "N/A") {
                            resultHtml += '<p>Sapling Score: ' + result.sapling_score + '</p>';
                        }
                        if (result.winston_score !== undefined && result.winston_score !== "N/A") {
                            resultHtml += '<p>Winston Score: ' + result.winston_score + '</p>';
                        }
                    } else if (result.model === "重写错误") {
                        resultHtml += '<p><b>原始句子:</b> ' + result.original_sentence + '</p>';
                        resultHtml += '<p class="text-danger"><b>错误:</b>' + result.rewritten_sentence + '</p>';
                         if (result.sapling_score !== undefined && result.sapling_score !== "N/A") {
                            resultHtml += '<p>Sapling Score: ' + result.sapling_score + '</p>';
                        }
                        if (result.winston_score !== undefined && result.winston_score !== "N/A") {
                            resultHtml += '<p>Winston Score: ' + result.winston_score + '</p>';
                        }
                    }
                    else {
                        // 显示最终结果
                        resultHtml += '<p>' + result.output + '</p>';
                        if (result.sapling_score !== undefined && result.sapling_score !== "N/A") {
                            resultHtml += '<p>Sapling Score: ' + result.sapling_score + '</p>';
                        }
                        if (result.winston_score !== undefined && result.winston_score !== "N/A") {
                            resultHtml += '<p>Winston Score: ' + result.winston_score + '</p>';
                        }
                        if (detector === "winston" && result.model ==="最终结果"){
                            resultHtml += '<p>消耗点数: ' + result.credits_used + '</p>';
                            resultHtml += '<p>可读性: ' + result.readability_score + '</p>';
                        }
                    }

                    resultHtml += '</div></div>';
                    $('#resultsContainer').append(resultHtml);
                });


            },
            error: function(xhr, status, error) {
              // 更详细的错误处理
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    if (xhr.responseJSON.details) {
                        $('#errorAlert').text(xhr.responseJSON.error + ": " + xhr.responseJSON.details).show();
                    } else {
                        $('#errorAlert').text(xhr.responseJSON.error).show();
                    }
                } else {
                    $('#errorAlert').text('请求失败: ' + error).show();
                }
                $('#currentSentenceCard').hide(); // 出错时也隐藏
            },
            complete: function() {
                // 隐藏加载动画
                $('#resultsLoading').hide();
                // 启用提交按钮
                $('#submitBtn').prop('disabled', false);
                $('#submitBtnText').show();
                $('#submitBtnSpinner').hide();

            }
        });
    });

    // 初始加载时获取一次模型列表,并设置AI检测器
    fetchGeminiModels();
    fetchSiliconFlowModels();
    $('#detectorSelect').trigger('change');
      // 更新加载状态的函数
    function updateLoadingStatus(stage, iteration, currentSentence) {
        var elapsedTime = (new Date().getTime() - startTime) / 1000; // 已用时间（秒）
        var estimatedTimePerIteration = elapsedTime / (iteration + 1); // 估计每次迭代的时间
        var estimatedRemainingTime = estimatedTimePerIteration * (estimatedTotalIterations - iteration -1); // 估计剩余时间
        if (estimatedRemainingTime < 0) {
            estimatedRemainingTime = 0; // 确保剩余时间不为负数
        }

        var statusText = stage + ' (第 ' + (iteration + 1) + ' 次迭代，预计剩余 ' + estimatedRemainingTime.toFixed(1) + ' 秒)';
        $('#loadingStatus').text(statusText);

        // 更新当前重写句子
        if (currentSentence) {
            $('#currentSentence').text(currentSentence);
        }
    }
    //导出函数供app.py调用。
    window.updateLoadingStatus = updateLoadingStatus;
});