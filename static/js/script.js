$(document).ready(function() {
    // 全局变量，用于存储开始时间
    var startTime;
    // 估计的总迭代次数 (可以根据经验调整)
    const estimatedTotalIterations = 5;

    // 获取 Gemini 模型列表的函数
    function fetchGeminiModels() {
        var geminiApiKey = $('#geminiApiKey').val();
        $('#geminiModelAlert').hide();

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
                $('#geminiModel').append($('<option>', {
                    value: "",
                    text: "选择模型"
                }));
                $.each(data.models, function(index, model) {
                    $('#geminiModel').append($('<option>', {
                        value: model,
                        text: model
                    }));
                });
                // 默认选中配置文件中的模型（如果存在）
                var configGeminiModel = "{{ config.get('MODELS', 'gemini', fallback='') }}";
                if (data.models.includes(configGeminiModel)) {
                    $('#geminiModel').val(configGeminiModel);
                }
                $('#geminiModelAlert').text('模型列表已更新。').show();
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
        $(this).find('i').addClass('fa-spin');
        fetchGeminiModels();
        $(this).find('i').removeClass('fa-spin');
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
                $('#siliconFlowModel').append($('<option>', {
                    value: "",
                    text: "选择模型"
                }));
                $.each(data.models, function(index, model) {
                    $('#siliconFlowModel').append($('<option>', {
                        value: model,
                        text: model
                    }));
                });
                // 默认选中配置文件中的模型（如果存在）
                var configSiliconFlowModel = "{{ config.get('MODELS', 'siliconflow', fallback='') }}";
                if (data.models.includes(configSiliconFlowModel)) {
                    $('#siliconFlowModel').val(configSiliconFlowModel);
                }
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

    // 改写模型选择改变事件
    $('#rewriteModelSelect').change(function() {
        var selectedModel = $(this).val();
        $('#geminiApiKeyGroup').hide();
        $('#geminiModelGroup').hide();
        $('#siliconFlowApiKeyGroup').hide();
        $('#siliconFlowModelGroup').hide();
        if (selectedModel === 'gemini') {
            $('#geminiApiKeyGroup').show();
            $('#geminiModelGroup').show();
            fetchGeminiModels(); // 自动刷新模型列表
        } else if (selectedModel === 'siliconflow') {
            $('#siliconFlowApiKeyGroup').show();
            $('#siliconFlowModelGroup').show();
            fetchSiliconFlowModels(); // 自动刷新模型列表
        }
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
        var selectedModel = $('#rewriteModelSelect').val();
        var geminiApiKey = $('#geminiApiKey').val();
        var siliconFlowApiKey = $('#siliconFlowApiKey').val();
        var detector = $('#detectorSelect').val();
        var saplingApiKey = $('#saplingApiKey').val();
        var winstonApiKey = $('#winstonApiKey').val();
        var geminiModel = $('#geminiModel').val();
        var siliconFlowModel = $('#siliconFlowModel').val();
        var sentenceThreshold = parseFloat($('#sentenceThreshold').val());
        var paragraphThreshold = parseFloat($('#paragraphThreshold').val());

        // 清空之前的错误信息和结果，并显示初始文本
        $('#errorAlert').hide();
        $('#resultsContainer').empty();
        $('#resultsContainer').append(`<div class="card mt-3"><div class="card-header"><b>原始文本</b></div><div class="card-body"><p id="originalText"></p></div></div><div class="card mt-3" id="currentSentenceCard" style="display: none;"><div class="card-header"><b>当前重写句子</b></div><div class="card-body"><p id="currentSentence"></p></div></div>`);
        $('#originalText').text(inputText);

        // 显示加载动画和初始状态
        $('#resultsLoading').show();
        updateLoadingStatus('正在准备', 0);

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

        // 根据选择的改写模型添加对应参数
        var requestData = {
            input_text: inputText,
            prompt: prompt,
            selected_model: selectedModel,
            detector: detector,
            ...apiKeyData,
            sentence_threshold: sentenceThreshold,
            paragraph_threshold: paragraphThreshold
        };
        if (selectedModel === 'gemini') {
            requestData.gemini_api_key = geminiApiKey;
            requestData.gemini_model = geminiModel;
        } else if (selectedModel === 'siliconflow') {
            requestData.siliconflow_api_key = siliconFlowApiKey;
            requestData.siliconflow_model = siliconFlowModel;
        }

        // 发送请求到后端
        $.ajax({
            url: '/process',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(requestData),
            success: function(data) {
                // 更新 API 用量和配额
                $('#geminiUsage').text(data.gemini_usage);
                $('#siliconFlowUsage').text(data.siliconflow_usage);
                if (detector === 'sapling') {
                    $('#saplingUsage').text(data.sapling_usage);
                } else {
                    $('#winstonBalance').text(data.winston_balance);
                }
                if (data.siliconflow_balance) {
                    $('#siliconFlowBalance').text(data.siliconflow_balance);
                }

                $('#currentSentenceCard').hide();

                // 显示结果
                $('#resultsContainer').empty();

                // 遍历 results 数组
                $.each(data.results, function(index, result) {
                    var resultHtml = '<div class="card mt-3">';
                    resultHtml += '<div class="card-header"><b>' + result.model + '</b></div>';
                    resultHtml += '<div class="card-body">';

                    if (result.model === "gemini" || result.model === "siliconflow" || result.model === "local") {
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
                    } else {
                        resultHtml += '<p>' + result.output + '</p>';
                        if (result.sapling_score !== undefined && result.sapling_score !== "N/A") {
                            resultHtml += '<p>Sapling Score: ' + result.sapling_score + '</p>';
                        }
                        if (result.winston_score !== undefined && result.winston_score !== "N/A") {
                            resultHtml += '<p>Winston Score: ' + result.winston_score + '</p>';
                        }
                        if (detector === "winston" && result.model === "最终结果") {
                            resultHtml += '<p>消耗点数: ' + result.credits_used + '</p>';
                            resultHtml += '<p>可读性: ' + result.readability_score + '</p>';
                        }
                    }

                    resultHtml += '</div></div>';
                    $('#resultsContainer').append(resultHtml);
                });
            },
            error: function(xhr, status, error) {
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    if (xhr.responseJSON.details) {
                        $('#errorAlert').text(xhr.responseJSON.error + ": " + xhr.responseJSON.details).show();
                    } else {
                        $('#errorAlert').text(xhr.responseJSON.error).show();
                    }
                } else {
                    $('#errorAlert').text('请求失败: ' + error).show();
                }
                $('#currentSentenceCard').hide();
            },
            complete: function() {
                $('#resultsLoading').hide();
                $('#submitBtn').prop('disabled', false);
                $('#submitBtnText').show();
                $('#submitBtnSpinner').hide();
            }
        });
    });

    // 初始加载时触发事件
    $('#rewriteModelSelect').trigger('change');
    $('#detectorSelect').trigger('change');

    // 更新加载状态的函数
    function updateLoadingStatus(stage, iteration, currentSentence) {
        var elapsedTime = (new Date().getTime() - startTime) / 1000;
        var estimatedTimePerIteration = elapsedTime / (iteration + 1);
        var estimatedRemainingTime = estimatedTimePerIteration * (estimatedTotalIterations - iteration - 1);
        if (estimatedRemainingTime < 0) {
            estimatedRemainingTime = 0;
        }

        var statusText = stage + ' (第 ' + (iteration + 1) + ' 次迭代，预计剩余 ' + estimatedRemainingTime.toFixed(1) + ' 秒)';
        $('#loadingStatus').text(statusText);

        if (currentSentence) {
            $('#currentSentence').text(currentSentence);
        }
    }
    window.updateLoadingStatus = updateLoadingStatus;
});