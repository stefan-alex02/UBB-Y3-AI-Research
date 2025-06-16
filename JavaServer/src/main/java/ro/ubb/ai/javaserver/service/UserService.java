package ro.ubb.ai.javaserver.service;

import ro.ubb.ai.javaserver.dto.user.RegisterRequest;
import ro.ubb.ai.javaserver.dto.user.UserDTO;
import ro.ubb.ai.javaserver.dto.user.UserUpdateRequest;

public interface UserService {
    UserDTO registerUser(RegisterRequest registerRequest);
    UserDTO getCurrentUser();
    UserDTO updateUser(Long userId, UserUpdateRequest updateRequest);
}
