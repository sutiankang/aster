# 数据身份、污染与敏感信息检查

`aster.data.contamination` 是原生、只读、本地文本审计器，不下载语料、不删除数据、不执行官方第三方流水线。它服务于训练/校准/评测共享的数据协议，**不是自动法律合规工具，也不能证明模型未见过某个未知的上游数据**。

## 固定输入

每行用 `TextSample(dataset_id, revision, split, sample_id, text)` 明确标识。`SplitManifest.from_samples(samples, *, source_uri, license_id=None, license_reference=None)` 固定一个划分的每个 sample ID 与原始文本 SHA256；正式审计时缺行、多行、重复 ID、换版本、换 split 或改文本都会拒绝。源地址应是稳定且无凭证的引用，拒绝用户名/密码和签名 query URL。划分应由调用方预先固定，不得审计失败后悄悄改测试全集。

`audit_text_splits(samples, manifests, *, config=ContaminationConfig(), allowed_licenses=(), pii_hmac_key=None)` 返回 `DataAuditReport`。`report.save(path)` 写入带内容哈希、不可覆盖的 JSON；`DataAuditReport.load(path)` 复核哈希。`corpus_fingerprint` 固定本次实际行内容；许可/来源另保留 manifest 指纹，不能用数据名称代替版本或内容校验。

## 精确与近似匹配

默认 `search='exhaustive'` 对预算内的所有文档对比较。精确匹配使用 NFKC、casefold 和空白合并后的完整文本。近似匹配使用字符 n-gram 集合的真实 Jaccard；中文和代码不依赖英文分词。保护样本全文嵌入长训练文档时，默认模式还有明确的子串检查，以免整体 Jaccard 很低而漏检。

大语料可以显式选择 `search='minhash_lsh'`：固定 seed 的 MinHash band 检索产生候选，再用真实 Jaccard 复核。规范化精确匹配在两种模式下都完整执行；**LSH 对近似文本和嵌入式测试片段可能漏检**。报告标记 `probabilistic_lsh_candidates`，不写成“全量无污染”。空文本独立报告，不把两个空集合的相似度当作 1。样本、字符、候选对和敏感信息命中都有上限，超限直接失败，不返回一个截断但声称完整的结果。

大小写/Unicode 规范化会改变某些代码语义，模板/样板文本也可能产生误报。因此 `training_exclusion_candidates` 只是训练侧待审查列表，绝不自动删除测试数据或修改输入。批准排除后须创建新的训练 manifest，并重新跑完整审计；不要靠临时过滤掩盖公开评测污染。

## 许可与敏感信息

许可清单只比较显式声明与宿主给出的 allowlist：`declared_allowlisted`、`declared_not_allowlisted`、`unknown_license`、`missing_reference`。不自动解释 SPDX 组合表达式，不推断网页作者授权，不检查合同/地域/用途等法律条件。allowlist 命中不是合法使用保证，必要时仍需人工核验来源条款。

PII/secret 规则覆盖邮箱、电话号码候选、若干令牌格式与私钥标记。报告只保存类型、位置和 HMAC 指纹，不包含匹配原文或上下文；同一私有 HMAC 密钥可关联 train/eval 的重复候选。默认密钥仅在本次审计内随机生成且不保存；需要可复现跨次关联时，由宿主秘密管理器提供相同的至少 32 字节密钥，**不得把密钥放进报告或版本库**。

规则会误报编号，也会漏掉姓名、地址、非标准密钥和间接身份；未命中不代表匿名化完成。文本 SHA、用户提供的 sample ID 和来源引用也可能具有关联或敏感性，报告应沿用原语料的访问权限，不能因未包含全文就公开发布。

## 与评测门禁组合

`data_audit_gate(report, *, corpus_fingerprint, require_exhaustive=True, allow_pii_candidates=False, allow_unreviewed_license=False)` 是工程准入条件：核对实际数据指纹，默认拒绝概率覆盖、跨 train/protected 边界重复、空样本、未审查 PII 和许可声明。显式放宽只改变工程策略，不生成法律许可或隐私保证。报告发现的命中、原始评测失败分母和模型质量报告应一起保留，不能拿这个门禁代替模型质量评测。

`tests/unit/test_contamination.py` 验证规范化精确重复、真实 MinHash 候选与独立 Jaccard、长文嵌入式测试片段、划分全集/内容篡改、PII 脱敏关联、许可声明状态、预算失败、确定性和报告完整性；测试文本是本地 fixture，不是公开语料污染率。

算法与工程划分参考 [Hugging Face Datatrove 的 MinHash 实现](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py) 的签名、分桶和候选处理思路，以及 [AllenAI Dolma 的数据处理工程](https://github.com/allenai/dolma)。本实现采用字符 shingle 与有界内存，不冒称它们的全分布式流水线、全套过滤器或检出率。
