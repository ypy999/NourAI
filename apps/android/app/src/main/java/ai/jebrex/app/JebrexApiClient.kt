package ai.jebrex.app

import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * Direct REST client connecting the Jebrex AI Android app strictly to a
 * user-configured local or LAN Jebrex AI backend (`api/backend.py`).
 * Never delegates responses to third-party cloud LLM APIs.
 */
class JebrexApiClient(private val baseUrl: String) {
    private var bearerToken: String? = null
    var activeSessionId: String? = null
        private set

    data class ChatReply(
        val reply: String,
        val responseSource: String,
        val generatedTokens: Int,
        val tokensPerSecond: Double,
        val sessionId: String
    )

    private fun requestJson(method: String, path: String, payload: JSONObject? = null): JSONObject {
        val url = URL("${baseUrl.trimEnd('/')}$path")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 8000
            readTimeout = 30000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            bearerToken?.let { setRequestProperty("Authorization", "Bearer $it") }
            if (payload != null) {
                doOutput = true
            }
        }
        if (payload != null) {
            OutputStreamWriter(conn.outputStream, Charsets.UTF_8).use { it.write(payload.toString()) }
        }
        val stream = if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
        val text = BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { it.readText() }
        if (conn.responseCode !in 200..299) {
            throw IllegalStateException("Jebrex API HTTP ${conn.responseCode}: $text")
        }
        return JSONObject(text)
    }

    fun authenticateGuest(): String {
        val res = requestJson("POST", "/api/v1/auth/guest", JSONObject())
        val token = res.getString("access_token")
        bearerToken = token
        return token
    }

    fun createSession(language: String = "arabic", title: String = "Jebrex Android"): String {
        if (bearerToken == null) authenticateGuest()
        val body = JSONObject().put("language", language).put("title", title)
        val res = requestJson("POST", "/api/v1/sessions", body)
        val sid = res.getJSONObject("session").getString("session_id")
        activeSessionId = sid
        return sid
    }

    fun sendMessage(
        message: String,
        language: String = "arabic",
        strategy: String = "greedy",
        maxNewTokens: Int = 96
    ): ChatReply {
        if (bearerToken == null) authenticateGuest()
        val sid = activeSessionId ?: createSession(language)
        val body = JSONObject()
            .put("session_id", sid)
            .put("message", message)
            .put("language", language)
            .put("strategy", strategy)
            .put("max_new_tokens", maxNewTokens)
            .put("use_fallback_assistant", false)
            .put("enable_web_search", false)

        val res = requestJson("POST", "/api/v1/chat", body)
        val metrics = res.optJSONObject("metrics") ?: JSONObject()
        return ChatReply(
            reply = res.optString("reply", ""),
            responseSource = res.optString("response_source", "neural_model"),
            generatedTokens = metrics.optInt("generated_tokens", 0),
            tokensPerSecond = metrics.optDouble("tokens_per_second", 0.0),
            sessionId = res.optString("session_id", sid)
        )
    }
}
